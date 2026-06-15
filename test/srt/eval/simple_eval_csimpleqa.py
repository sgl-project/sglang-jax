# Adapted from https://github.com/openai/simple-evals/

"""
Chinese-SimpleQA: factual QA in Chinese.
Dataset: OpenStellarTeam/Chinese-SimpleQA (3000 train rows).

GRADER_TEMPLATE is copied verbatim from the official evaluator at
https://github.com/LivingFutureLab/ChineseSimpleQA/blob/main/chinese_simpleqa_eval.py
and produces A (correct) / B (incorrect) / C (not attempted) labels. The
official setup recommends GPT-4o as the grader; passing the same served
model as the grader is biased but lands within ~1pp on Ling-2.6-Flash.
"""

import re

import eval.simple_eval_common as common
from eval.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

GRADER_TEMPLATE = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如："可能是"，"虽然我不确定，但我认为"），也视为错误。

以下是【未尝试】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题"金山铁路黄浦江特大桥的全长是多少米？"，标准答案为"3518.17"：
    - 预测答案"3518"、"3518.1"、"3518.17"均为【正确】。
    - 预测答案"3520"和"3600"均为【错误】。
    - 预测答案"大约3500米"和"超过3000米"被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题"菱镁矿的主要化学成分是什么？"标准答案为"碳酸镁（MgCO3）"。"碳酸镁"或"MgCO3"均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题"巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？"标准答案为"意大利撒丁岛"，预测答案"撒丁岛"被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是"Robinson"，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {target}
预测答案: {predicted_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()


def parse_grade(text: str) -> str:
    # Match official: re.search(r"(A|B|C)", grading_response); default "C".
    match = re.search(r"(A|B|C)", (text or "").upper())
    return match.group(0) if match else "C"


class ChineseSimpleQAEval(Eval):
    def __init__(
        self,
        grader: SamplerBase,
        num_examples: int | None = None,
        num_threads: int = 64,
    ):
        from datasets import load_dataset

        dataset = load_dataset("OpenStellarTeam/Chinese-SimpleQA", split="train")
        examples = [{"question": row["question"], "answer": str(row["answer"])} for row in dataset]
        if num_examples:
            examples = examples[: min(num_examples, len(examples))]
        self.examples = examples
        self.grader = grader
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            # Official answer pass: raw question only, no system prompt.
            prompt_messages = [sampler._pack_message(content=row["question"], role="user")]
            response_text = sampler(prompt_messages) or ""

            grader_prompt = GRADER_TEMPLATE.format(
                question=row["question"],
                target=row["answer"],
                predicted_answer=response_text,
            )
            grader_text = self.grader(
                [self.grader._pack_message(content=grader_prompt, role="user")]
            )
            grade = parse_grade(grader_text)

            is_correct = grade == "A"
            is_incorrect = grade == "B"
            is_not_attempted = grade == "C"
            score = 1.0 if is_correct else 0.0

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=grade,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "is_correct": float(is_correct),
                    "is_incorrect": float(is_incorrect),
                    "is_not_attempted": float(is_not_attempted),
                },
            )

        results = common.map_with_progress(fn, self.examples, num_threads=self.num_threads)
        return common.aggregate_results(results)
