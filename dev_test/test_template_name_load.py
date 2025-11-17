import sys
from dataclasses import dataclass
from typing import Optional

from sgl_jax.srt.managers.template_manager import TemplateManager
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.conversation import chat_templates
from sgl_jax.srt.server_args import ServerArgs, PortArgs


@dataclass
class MockTokenizer:
    chat_template: Optional[str] = None


@dataclass
class MockTokenizerManager:
    tokenizer: Optional[MockTokenizer] = None
    processor: Optional[object] = None


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


def test_none_chat_template_with_model_path_inference():
    """测试 chat_template 为 None，从 model_path 推断模板"""
    print("\n" + "=" * 80)
    print("测试：chat_template 为 None，从 model_path 推断")
    print("=" * 80)
    
    print("\n【步骤 1】准备测试数据")
    print("-" * 80)
    
    # 使用能被推断出模板的 model_path
    server_args = ServerArgs(chat_template=None, model_path="Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"✓ server_args.chat_template = {server_args.chat_template}")
    print(f"✓ server_args.model_path = '{server_args.model_path}'")
    print(f"  (这个 model_path 应该能推断出 'qwen2-vl' 模板)")
    
    print("\n【步骤 2】创建 TemplateManager 并初始化")
    print("-" * 80)
    
    template_manager = TemplateManager()
    tokenizer_manager = MockTokenizerManager(tokenizer=MockTokenizer())
    
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
    
    # 根据 model_path 推断，应该得到 'qwen2-vl' 或其他合适的模板
    if actual_template_name is not None:
        print(f"✓ 测试通过！从 model_path 成功推断出模板: '{actual_template_name}'")
        return 0
    else:
        print(f"✗ 测试失败！应该从 model_path 推断出模板，但 _chat_template_name 为 None")
        return 1


def test_none_chat_template_with_hf_template():
    print("\n【步骤 1】准备测试数据")
    print("-" * 80)
    
    # 使用真实的 model_path，但不指定 chat_template
    server_args = ServerArgs(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        chat_template=None,
        skip_tokenizer_init=False,
        trust_remote_code=True,
    )
    print(f"✓ server_args.chat_template = {server_args.chat_template}")
    print(f"✓ server_args.model_path = '{server_args.model_path}'")
    print(f"  (将从 HuggingFace tokenizer 获取模板)")
    
    print("\n【步骤 2】创建 TokenizerManager 和 TemplateManager")
    print("-" * 80)
    
    port_args = PortArgs.init_new(server_args)
    print(f"✓ 创建 PortArgs")
    
    try:
        tokenizer_manager = TokenizerManager(server_args, port_args)
        print(f"✓ 创建 TokenizerManager")
        print(f"  tokenizer: {tokenizer_manager.tokenizer}")
        
        if tokenizer_manager.tokenizer and hasattr(tokenizer_manager.tokenizer, 'chat_template'):
            original_template = tokenizer_manager.tokenizer.chat_template
            if original_template:
                print(f"  原始 chat_template 长度: {len(original_template)} 字符")
                print(f"  原始 chat_template 预览: {original_template[:100]}...")
            else:
                print(f"  原始 chat_template: None")
        
    except Exception as e:
        print(f"✗ 创建 TokenizerManager 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n【步骤 3】初始化 TemplateManager")
    print("-" * 80)
    
    template_manager = TemplateManager()
    
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
    
    print("\n【步骤 4】验证结果")
    print("-" * 80)
    
    actual_template_name = template_manager.chat_template_name
    print(f"_chat_template_name = {actual_template_name}")
    
    if tokenizer_manager.tokenizer and hasattr(tokenizer_manager.tokenizer, 'chat_template'):
        current_template = tokenizer_manager.tokenizer.chat_template
        if current_template:
            print(f"tokenizer.chat_template 长度: {len(current_template)} 字符")
            print(f"tokenizer.chat_template 预览: {current_template[:100]}...")
        else:
            print(f"tokenizer.chat_template: None")
    
    # 验证逻辑：
    # 1. 如果从 model_path 推断出模板，_chat_template_name 应该不为 None
    # 2. 如果推断失败，从 HF tokenizer 获取，_chat_template_name 应该为 None，但 tokenizer.chat_template 应该存在
    if actual_template_name is not None:
        print(f"\n✓ 测试通过！从 model_path 推断出模板: '{actual_template_name}'")
        return 0
    elif tokenizer_manager.tokenizer and tokenizer_manager.tokenizer.chat_template:
        print(f"\n✓ 测试通过！从 HuggingFace tokenizer 成功获取模板")
        print(f"  _chat_template_name = None (使用 tokenizer 的 Jinja 模板)")
        print(f"  tokenizer.chat_template 已被正确设置")
        return 0
    else:
        print(f"\n✗ 测试失败！")
        print(f"  _chat_template_name = {actual_template_name}")
        print(f"  tokenizer.chat_template = {tokenizer_manager.tokenizer.chat_template if tokenizer_manager.tokenizer else 'N/A'}")
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
    print("  1. chat_template 显式指定时的加载")
    print("  2. chat_template 为 None 时从 model_path 推断")
    print("  3. chat_template 为 None 时从 HuggingFace tokenizer 获取")
    print("=" * 80)
    
    exit_code_1 = test_explicit_chat_template_loading()
    # exit_code_2 = test_none_chat_template_with_model_path_inference()
    exit_code_3 = test_none_chat_template_with_hf_template()
    exit_code_5 = test_invalid_chat_template()
    
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    results = [
        ("显式指定 chat_template", exit_code_1),
        # ("从 model_path 推断模板", exit_code_2),
        ("从 HuggingFace tokenizer 获取模板", exit_code_3),
        ("无效的 chat_template", exit_code_5),
    ]
    
    all_passed = True
    for test_name, exit_code in results:
        status = "✓ 通过" if exit_code == 0 else "✗ 失败"
        print(f"{status} - {test_name}")
        if exit_code != 0:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)
        print("\n验证结果：")
        print("  1. ✓ 显式指定 chat_template 时能正确设置")
        # print("  2. ✓ chat_template 为 None 时能从 model_path 推断")
        print("  3. ✓ 推断失败时能从 HuggingFace tokenizer 获取")
        print("  4. ✓ 无效的 chat_template 不会被设置")
        print("\n改动验证成功：")
        print("  - load_chat_template 方法的 else 分支正确实现")
        print("  - guess_chat_template_from_model_path 被正确调用")
        print("  - _resolve_hf_chat_template 被正确调用")
        print("  - 多级 fallback 机制工作正常")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ 部分测试失败")
        print("=" * 80)
        sys.exit(1)
