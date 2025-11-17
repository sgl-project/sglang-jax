"""
测试 chat_template 加载流程

Engine.__init__->_launch_subprocesses->initialize_templates->load_chat_template
1. ->_load_explicit_chat_template: load from server_args.chat_template
2. 

测试目标：
1. 传入 server_args.chat_template = "qwen2-vl"
2. 验证 TemplateManager._chat_template_name 被正确设置为 "qwen2-vl"
"""

import sys
from dataclasses import dataclass
from typing import Optional

from sgl_jax.srt.managers.template_manager import TemplateManager
from sgl_jax.srt.conversation import chat_templates


@dataclass
class MockTokenizerManager:
    pass


@dataclass
class ServerArgs:
    chat_template: Optional[str] = None
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct"


def test_explicit_chat_template_loading():
    # 步骤 1: 准备测试数据
    server_args = ServerArgs(chat_template="qwen2-vl")
    
    # 检查模板是否已注册
    print(f"\n可用的 chat templates: {list(chat_templates.keys())}")
    
    if server_args.chat_template not in chat_templates:
        raise Exception(f"✗ 失败: 模板 '{server_args.chat_template}' 未注册")
    
    # 步骤 2: 创建 TemplateManager
    template_manager = TemplateManager()
    print(f"  初始 _chat_template_name: {template_manager.chat_template_name}")
    
    # 步骤 3: 模拟初始化流程
    tokenizer_manager = MockTokenizerManager()
    print(f"✓ 创建 MockTokenizerManager")
    
    # 调用 initialize_templates (模拟 Engine._launch_subprocesses 中的调用)
    print(f"\n调用: template_manager.initialize_templates(")
    print(f"    tokenizer_manager={tokenizer_manager},")
    print(f"    model_path='{server_args.model_path}',")
    print(f"    chat_template='{server_args.chat_template}'")
    print(f")")
    
    try:
        template_manager.initialize_templates(
            tokenizer_manager=tokenizer_manager,
            model_path=server_args.model_path,
            chat_template=server_args.chat_template
        )
        print("✓ initialize_templates 调用成功")
    except Exception as e:
        print(f"✗ 失败: initialize_templates 调用出错 - {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 步骤 4: 验证结果
    actual_template_name = template_manager.chat_template_name
    expected_template_name = server_args.chat_template
    
    print(f"预期值: '{expected_template_name}'")
    print(f"实际值: '{actual_template_name}'")
    
    if actual_template_name == expected_template_name:
        print(f"✓ 测试通过！_chat_template_name 正确设置为 '{actual_template_name}'")
        return 0
    else:
        print(f"✗ 测试失败！预期 '{expected_template_name}'，实际 '{actual_template_name}'")
        return 1


def test_none_chat_template():
    """测试 chat_template 为 None 的情况"""
    print("\n" + "=" * 80)
    print("测试：chat_template 为 None 的情况")
    print("=" * 80)
    
    print("\n【步骤 1】准备测试数据")
    print("-" * 80)
    
    server_args = ServerArgs(chat_template=None)
    print(f"✓ server_args.chat_template = {server_args.chat_template}")
    
    print("\n【步骤 2】创建 TemplateManager 并初始化")
    print("-" * 80)
    
    template_manager = TemplateManager()
    tokenizer_manager = MockTokenizerManager()
    
    try:
        template_manager.initialize_templates(
            tokenizer_manager=tokenizer_manager,
            model_path=server_args.model_path,
            chat_template=server_args.chat_template
        )
        print("✓ initialize_templates 调用成功")
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n【步骤 3】验证结果")
    print("-" * 80)
    
    actual_template_name = template_manager.chat_template_name
    print(f"_chat_template_name = {actual_template_name}")
    
    if actual_template_name is None:
        print("✓ 测试通过！chat_template 为 None 时，_chat_template_name 保持为 None")
        return 0
    else:
        print(f"✗ 测试失败！预期 None，实际 '{actual_template_name}'")
        return 1


def test_invalid_chat_template():
    """测试无效的 chat_template"""
    print("\n" + "=" * 80)
    print("测试：无效的 chat_template")
    print("=" * 80)
    
    print("\n【步骤 1】准备测试数据")
    print("-" * 80)
    
    invalid_template = "invalid-template-name"
    server_args = ServerArgs(chat_template=invalid_template)
    print(f"✓ server_args.chat_template = '{server_args.chat_template}'")
    print(f"  (这是一个不存在的模板名称)")
    
    print("\n【步骤 2】创建 TemplateManager 并初始化")
    print("-" * 80)
    
    template_manager = TemplateManager()
    tokenizer_manager = MockTokenizerManager()
    
    try:
        template_manager.initialize_templates(
            tokenizer_manager=tokenizer_manager,
            model_path=server_args.model_path,
            chat_template=server_args.chat_template
        )
        print("✓ initialize_templates 调用成功（未抛出异常）")
    except Exception as e:
        print(f"✗ 抛出异常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n【步骤 3】验证结果")
    print("-" * 80)
    
    actual_template_name = template_manager.chat_template_name
    print(f"_chat_template_name = {actual_template_name}")
    
    if actual_template_name is None:
        print("✓ 测试通过！无效的模板名称不会被设置，_chat_template_name 保持为 None")
        return 0
    else:
        print(f"✗ 测试失败！预期 None，实际 '{actual_template_name}'")
        return 1


if __name__ == "__main__":
    exit_code_1 = test_explicit_chat_template_loading()
    exit_code_2 = test_none_chat_template()
    exit_code_3 = test_invalid_chat_template()
    
    results = [
        ("显式指定 chat_template", exit_code_1),
        ("chat_template 为 None", exit_code_2),
        ("无效的 chat_template", exit_code_3),
    ]
    
    all_passed = True
    for test_name, exit_code in results:
        status = "✓ 通过" if exit_code == 0 else "✗ 失败"
        print(f"{status} - {test_name}")
        if exit_code != 0:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n✓ 所有测试通过！")
        print("\n验证结果：")
        print("  1. ✓ 通过 server_args.chat_template 传入的模板名称能正确设置")
        print("  2. ✓ _chat_template_name 被正确赋值")
        print("  3. ✓ 加载流程符合预期")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败")
        sys.exit(1)
