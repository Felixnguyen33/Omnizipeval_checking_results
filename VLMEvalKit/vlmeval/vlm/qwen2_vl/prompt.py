from __future__ import annotations

import json
import re

# Qwen2.5-VL technical report (arXiv:2502.13923) ScreenSpot / ScreenSpot-Pro protocol.
# Official cookbook: https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb
# min/max_pixels match the cookbook defaults (4 and 16384 visual tokens).
QWEN25_SCREENSPOT_MIN_PIXELS = 4 * 28 * 28  # 3136
QWEN25_SCREENSPOT_MAX_PIXELS = 16384 * 28 * 28  # 12845056
QWEN25_SCREENSPOT_DISPLAY = 1000  # cookbook ComputerUse display_width/height_px

_SCREENSPOT_USER_PREFIXES = (
    'Please complete the following tasks by clicking using `pyautogui.click`:\n',
    'Please click the following target element using `pyautogui.click`:\n',
)


def is_qwen25_vl_path(model_path: str | None) -> bool:
    path = (model_path or '').lower().replace('_', '-')
    return 'qwen2.5-vl' in path or 'qwen2-5-vl' in path


def is_screenspot_dataset(dataset: str | None) -> bool:
    return dataset is not None and 'ScreenSpot' in str(dataset)


def extract_screenspot_instruction(message: list[dict]) -> str:
    texts = [m.get('value', '') for m in message if m.get('type') == 'text']
    instruction = texts[-1] if texts else ''
    for prefix in _SCREENSPOT_USER_PREFIXES:
        if instruction.startswith(prefix):
            return instruction[len(prefix):]
    return instruction


def qwen25_computer_use_system_prompt(display_px: int = QWEN25_SCREENSPOT_DISPLAY) -> str:
    """NousFnCall-style computer_use tool prompt from the Qwen2.5-VL cookbook."""
    computer_use = {
        'type': 'function',
        'function': {
            'name': 'computer_use',
            'description': (
                'Use a mouse and keyboard to interact with a computer, and take screenshots.\n'
                '* This is an interface to a desktop GUI. You do not have access to a terminal '
                'or applications menu. You must click on desktop icons to start applications.\n'
                '* Some applications may take time to start or process actions, so you may need '
                'to wait and take successive screenshots to see the results of your actions. '
                'E.g. if you click on Firefox and a window doesn\'t open, try wait and taking '
                'another screenshot.\n'
                f'* The screen\'s resolution is {display_px}x{display_px}.\n'
                '* Whenever you intend to move the cursor to click on an element like an icon, '
                'you should consult a screenshot to determine the coordinates of the element '
                'before moving the cursor.\n'
                '* If you tried clicking on a program or link but it failed to load, even after '
                'waiting, try adjusting your cursor position so that the tip of the cursor '
                'visually falls on the element that you want to click.\n'
                '* Make sure to click any buttons, links, icons, etc with the cursor tip in the '
                'center of the element. Don\'t click boxes on their edges unless asked.'
            ),
            'parameters': {
                'properties': {
                    'action': {
                        'description': (
                            'The action to perform. The available actions are:\n'
                            '* `key`: Performs key down presses on the arguments passed in order, '
                            'then performs key releases in reverse order.\n'
                            '* `type`: Type a string of text on the keyboard.\n'
                            '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                            '* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n'
                            '* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n'
                            '* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n'
                            '* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n'
                            '* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n'
                            '* `scroll`: Performs a scroll of the mouse scroll wheel.\n'
                            '* `wait`: Wait specified seconds for the change to happen.\n'
                            '* `terminate`: Terminate the current task and report its completion status.'
                        ),
                        'enum': [
                            'key', 'type', 'mouse_move', 'left_click', 'left_click_drag',
                            'right_click', 'middle_click', 'double_click', 'scroll', 'wait',
                            'terminate',
                        ],
                        'type': 'string',
                    },
                    'coordinate': {
                        'description': (
                            '(x, y): The x (pixels from the left edge) and y (pixels from the '
                            'top edge) coordinates to move the mouse to.'
                        ),
                        'type': 'array',
                    },
                    'text': {
                        'description': 'Required only by `action=type`.',
                        'type': 'string',
                    },
                    'keys': {
                        'description': 'Required only by `action=key`.',
                        'type': 'array',
                    },
                    'time': {
                        'description': 'The seconds to wait. Required only by `action=wait`.',
                        'type': 'number',
                    },
                    'status': {
                        'description': 'The status of the task. Required only by `action=terminate`.',
                        'type': 'string',
                        'enum': ['success', 'failure'],
                    },
                },
                'required': ['action'],
                'type': 'object',
            },
        },
    }
    tool_json = json.dumps(computer_use, ensure_ascii=False)
    return (
        'You are a helpful assistant.\n\n# Tools\n\n'
        'You may call one or more functions to assist with the user query.\n\n'
        'You are provided with function signatures within <tools></tools> XML tags:\n'
        f'<tools>\n{tool_json}\n</tools>\n\n'
        'For each function call, return a json object with function name and arguments '
        'within <tool_call></tool_call> XML tags:\n'
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def computer_use_response_to_pyautogui(response: str) -> str:
    """Map Qwen computer_use coordinates (0-1000) to VLMEvalKit's pyautogui parser."""
    match = re.search(
        r'"coordinate"\s*:\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]',
        response,
    )
    if match is None:
        match = re.search(r'x\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([+-]?\d+(?:\.\d+)?)', response)
    if match is None:
        return response
    x, y = float(match.group(1)), float(match.group(2))
    if x > 1 or y > 1:
        x = x / QWEN25_SCREENSPOT_DISPLAY
        y = y / QWEN25_SCREENSPOT_DISPLAY
    return f'pyautogui.click(x={x}, y={y})'


class Qwen2VLPromptMixin:
    """
    Mixin class for Qwen2VLChat to build custom prompt for different datasets.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            if dataset is not None and 'LEGO' in dataset:
                return False
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:  # MME has it's own prompt
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet', 'ChartQAPro', 'ChartQAPro_CoT', 'ChartQAPro_PoT', 'ChartMuseum'}:  # noqa: E501
            return True
        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""

        import string

        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'

        import string

        import pandas as pd

        def cn_string(s):
            import re

            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for YORN dataset:"""
        YORN_PROMPT = ' Please answer yes or no.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs
