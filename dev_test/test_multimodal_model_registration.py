import sys

from sgl_jax.srt.configs.model_config import ModelConfig, is_multimodal_model, multimodal_model_archs


def test_real_model_configs():

    all_passed = True
    
    print("\n【测试 1】Qwen/Qwen2.5-VL-7B-Instruct (预期: is_multimodal=True)")
    print("-" * 80)
    
    try:
        model_config = ModelConfig(
            model_path="Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        print(f"模型路径: {model_config.model_path}")
        print(f"模型架构: {model_config.hf_config.architectures}")
        print(f"is_multimodal: {model_config.is_multimodal}")
        
        if model_config.is_multimodal:
            print("✓ 通过: 正确识别为多模态模型")
        else:
            print("✗ 失败: 应该识别为多模态模型，但返回 False")
            all_passed = False
            
    except Exception as e:
        print(f"✗ 错误: 加载模型配置失败 - {e}")
        all_passed = False
    
    print("\n【测试 2】Qwen/Qwen2.5-7B-Instruct (预期: is_multimodal=False)")
    print("-" * 80)
    
    try:
        model_config = ModelConfig(
            model_path="Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True
        )
        
        print(f"模型路径: {model_config.model_path}")
        print(f"模型架构: {model_config.hf_config.architectures}")
        print(f"is_multimodal: {model_config.is_multimodal}")
        
        if not model_config.is_multimodal:
            print("✓ 通过: 正确识别为非多模态模型")
        else:
            print("✗ 失败: 应该识别为非多模态模型，但返回 True")
            all_passed = False
            
    except Exception as e:
        print(f"✗ 错误: 加载模型配置失败 - {e}")
        all_passed = False
    
    # 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有真实模型测试通过！")
        print("=" * 80)
        return 0
    else:
        print("✗ 部分真实模型测试失败！")
        print("=" * 80)
        return 1


def test_multimodal_model_detection():
    
    print("\n【测试 1】测试已知的多模态模型架构")
    print("-" * 80)
    
    test_cases_multimodal = [
        ["Qwen2VLForConditionalGeneration"],
        ["Qwen2_5_VLForConditionalGeneration"],
        ["LlavaForConditionalGeneration"],
    ]
    
    all_passed = True
    for arch_list in test_cases_multimodal:
        result = is_multimodal_model(arch_list)
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {arch_list[0]} -> {result}")
        if not result:
            all_passed = False
    
    print("\n【测试 2】测试非多模态模型架构")
    print("-" * 80)
    
    test_cases_non_multimodal = [
        ["Qwen2ForCausalLM"],
        ["LlamaForCausalLM"],
        ["GPT2LMHeadModel"],
    ]
    
    for arch_list in test_cases_non_multimodal:
        result = is_multimodal_model(arch_list)
        status = "✓ 通过" if not result else "✗ 失败"
        print(f"{status}: {arch_list[0]} -> {result}")
        if result:
            all_passed = False
    
    print("\n【信息】当前支持的多模态模型架构列表")
    print("-" * 80)
    print(f"共 {len(multimodal_model_archs)} 个多模态架构:")
    for i, arch in enumerate(multimodal_model_archs, 1):
        print(f"  {i:2d}. {arch}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有架构级别测试通过！")
        print("=" * 80)
        return 0
    else:
        print("✗ 部分架构级别测试失败！")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    
    exit_code_1 = test_real_model_configs()
    exit_code_2 = test_multimodal_model_detection()
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if exit_code_1 == 0 and exit_code_2 == 0:
        print("✓ 所有测试套件通过！")
        print("\n改动验证成功：")
        print("  1. ✓ Qwen/Qwen2.5-VL-7B-Instruct 正确识别为多模态模型")
        print("  2. ✓ Qwen/Qwen2.5-7B-Instruct 正确识别为非多模态模型")
        print("  3. ✓ is_multimodal_model 函数正确实现")
        print("  4. ✓ 已成功集成到 ModelConfig 类中")
        sys.exit(0)
    else:
        print("✗ 存在测试失败")
        sys.exit(1)
