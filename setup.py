from setuptools import setup, find_packages

setup(
    name="rhen",
    version="1.0.0",
    description="RHEN: Non-Autoregressive JEV-Type Decision Engine for Agentic AI",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    py_modules=[
        "adapter",
        "agent_tools",
        "config",
        "context_builder",
        "eos_eval",
        "eos_loop",
        "meta_cognitive_monitor",
        "model",
        "optimizers",
        "rhen_chunker",
        "rhen_cognition",
        "rhen_credit",
        "rhen_loss",
        "rhen_packer",
        "rhen_pillars",
        "rhen_verify",
        "tokenizer",
        "unified_orchestrator",
    ],
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0,<6.0.0",
        "numpy>=1.24.0",
    ],
    package_data={
        "rhen": ["py.typed", "tokenizer_assets/*"],
        "*": ["EOS_Tokenizer/*", "configs/*.json"],
    },
    include_package_data=True,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "rhen-decide=cli.decide:main",
            "rhen-train=cli.train:main",
            "rhen-eval=cli.eval:main",
            "rhen-tasks=cli.tasks:main",
        ],
    },
)
