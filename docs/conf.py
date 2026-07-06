project = "SGL-JAX"
author = "SGL-JAX contributors"
copyright = "2026, SGL-JAX contributors"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "cookbook/**",
    "design/**",
    "research/**",
    "superpowers/**",
    "developer_guide/qwen3_omni_v6e16_deployment.md",
    "developer_guide/sjmrt_v6e16_validation_log.md",
    "gla_radix_linear_attention_alignment.md",
    "kimi_linear_architecture_analysis.md",
    "lightning_backend_dp_refactor.md",
    "ling_2_6_gla_pr_section.md",
    "mimo_v2_flash_eval_guide.md",
    "mimo_v2_flash_eval_ops.md",
    "mini_eval_grading_rules.md",
    "sampler_comparison_report.md",
    "swa_mempool_design.md",
    "swa_radix_cache_e2e_test_report.md",
]

suppress_warnings = [
    "misc.highlighting_failure",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

html_theme = "sphinx_book_theme"
html_title = "SGL-JAX Documentation"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/sgl-project/sglang-jax",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
