"""
- _apply_conversation_template()
    -> generate_chat_conv()
    -> conv.get_prompt()
"""
import sys
from typing import List, Optional, Any
from dataclasses import dataclass

from sgl_jax.srt.conversation import generate_chat_conv, chat_templates


@dataclass
class ImageURL:
    url: str
    detail: str = "auto"


@dataclass
class ContentPart:
    """消息内容部分"""
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None
    modalities: str = "image"


@dataclass
class Message:
    role: str
    content: Any  # str or List[ContentPart]


@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[Message]
    continue_final_message: bool = False


class MockMessage:
    def __init__(self, role: str, content: Any):
        self.role = role
        self.content = content


class MockTextContent:
    """模拟文本内容对象"""
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockImageUrl:
    """模拟图片 URL 对象"""
    def __init__(self, url: str, detail: str = "auto"):
        self.url = url
        self.detail = detail


class MockImageContent:
    """模拟图片内容对象"""
    def __init__(self, url: str, detail: str = "auto"):
        self.type = "image_url"
        self.image_url = MockImageUrl(url, detail)
        self.modalities = "image"


def test_qwen2_vl_template():    
    print("\n【步骤 1】检查 qwen2-vl 模板是否已注册")
    print("-" * 80)
    
    if "qwen2-vl" not in chat_templates:
        raise Exception("✗ 失败: qwen2-vl 模板未注册")
    
    print("✓ qwen2-vl 模板已注册")
    template = chat_templates["qwen2-vl"]
    print(f"  - 模板名称: {template.name}")
    print(f"  - 系统消息: {template.system_message}")
    print(f"  - 图片 token: {template.image_token}")
    print(f"  - 视频 token: {template.video_token}")
    
    print("\n【步骤 2】构造测试消息")
    print("-" * 80)
    
    messages = [
        MockMessage(
            role="user",
            content=[
                MockTextContent(text="描述这张图片"),
                MockImageContent(url="https://github.com/sgl-project/sglang-jax/blob/main/test/srt/example_image.png?raw=true")
            ]
        )
    ]
    
    print("消息内容:")
    print("  - role: user")
    print("  - content:")
    print("    - type: text, text: '描述这张图片'")
    print("    - type: image_url, url: 'https://github.com/sgl-project/sglang-jax/blob/main/test/srt/example_image.png?raw=true'")
    
    print("\n【步骤 3】生成 conversation")
    print("-" * 80)
    
    try:
        conv = generate_chat_conv(messages, "qwen2-vl")
        print("✓ 成功生成 conversation 对象")
    except Exception as e:
        print(f"✗ 失败: 生成 conversation 时出错 - {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n【步骤 4】获取生成的 prompt")
    print("-" * 80)
    
    try:
        prompt = conv.get_prompt()
        print("✓ 成功获取 prompt")
        print("\n生成的 prompt:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
    except Exception as e:
        print(f"✗ 失败: 获取 prompt 时出错 - {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n【步骤 5】验证 prompt 内容")
    print("-" * 80)
    
    expected_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
描述这张图片<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant
"""
    
    all_checks_passed = True
    
    # 检查关键元素
    checks = [
        ("<|im_start|>system", "系统消息开始标记"),
        ("You are a helpful assistant.", "系统消息内容"),
        ("<|im_end|>", "消息结束标记"),
        ("<|im_start|>user", "用户消息开始标记"),
        ("描述这张图片", "用户文本内容"),
        ("<|vision_start|><|image_pad|><|vision_end|>", "图片 token"),
        ("<|im_start|>assistant", "助手消息开始标记"),
    ]
    
    for check_str, description in checks:
        if check_str in prompt:
            print(f"✓ 包含 {description}: '{check_str}'")
        else:
            print(f"✗ 缺失 {description}: '{check_str}'")
            all_checks_passed = False
    
    # 检查完整匹配
    print("\n【步骤 6】检查完整匹配")
    print("-" * 80)
    
    if prompt == expected_prompt:
        print("✓ 生成的 prompt 与预期完全匹配！")
    else:
        print("✗ 生成的 prompt 与预期不完全匹配")
        print("\n预期 prompt:")
        print("-" * 80)
        print(repr(expected_prompt))
        print("-" * 80)
        print("\n实际 prompt:")
        print("-" * 80)
        print(repr(prompt))
        print("-" * 80)
        
        # 显示差异
        print("\n差异分析:")
        if len(prompt) != len(expected_prompt):
            print(f"  - 长度不同: 预期 {len(expected_prompt)}, 实际 {len(expected_prompt)}")
        
        # 逐行比较
        expected_lines = expected_prompt.split('\n')
        actual_lines = prompt.split('\n')
        
        for i, (exp_line, act_line) in enumerate(zip(expected_lines, actual_lines)):
            if exp_line != act_line:
                print(f"  - 第 {i+1} 行不同:")
                print(f"    预期: {repr(exp_line)}")
                print(f"    实际: {repr(act_line)}")
        
        all_checks_passed = False
    
    print("\n" + "=" * 80)
    if all_checks_passed and prompt == expected_prompt:
        print("✓ 所有测试通过！")
        print("=" * 80)
        return 0
    else:
        print("✗ 部分测试失败")
        print("=" * 80)
        return 1


def test_different_chat_templates():
    """测试不同的 chat template"""
    
    print("\n" + "=" * 80)
    print("测试不同 Chat Template 的输出")
    print("=" * 80)
    
    # 测试消息
    messages = [
        MockMessage(
            role="user",
            content=[
                MockTextContent(text="描述这张图片"),
                MockImageContent(url="https://github.com/sgl-project/sglang-jax/blob/main/test/srt/example_image.png?raw=true")
            ]
        )
    ]
    
    # 获取所有可用的模板
    available_templates = list(chat_templates.keys())
    print(f"\n可用的 chat templates: {available_templates}")
    
    # 测试每个模板
    for template_name in available_templates:
        print(f"\n【测试模板】{template_name}")
        print("-" * 80)
        
        try:
            conv = generate_chat_conv(messages, template_name)
            prompt = conv.get_prompt()
            
            print(f"✓ 成功生成 prompt (长度: {len(prompt)} 字符)")
            print("生成的 prompt 预览 (前 200 字符):")
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
            
        except Exception as e:
            print(f"✗ 失败: {e}")
    
    return 0


if __name__ == "__main__":    
    exit_code_1 = test_qwen2_vl_template()
    exit_code_2 = test_different_chat_templates()
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if exit_code_1 == 0:
        print("✓ qwen2-vl 模板测试通过！")
        print("\n改动验证成功：")
        print("  1. ✓ qwen2-vl 模板已正确注册")
        print("  2. ✓ 能够正确处理多模态消息（文本 + 图片）")
        print("  3. ✓ 生成的 prompt 格式正确")
        print("  4. ✓ 图片 token 位置正确")
        sys.exit(0)
    else:
        print("✗ 存在测试失败")
        sys.exit(1)
