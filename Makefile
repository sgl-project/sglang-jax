.PHONY: dev check-zsh install-omz install-brew install-plugins setup-zshrc setup-ssh setup-conda help

# Development environment setup for Linux
# Default target
help:
	@echo "Available commands:"
	@echo "  make dev - Install development environment (all plugins and configs)"

# Main development environment setup command
dev: check-zsh install-omz install-brew install-plugins setup-zshrc setup-ssh setup-conda
	@echo "Development environment setup complete!"
	@echo "Please run 'source ~/.zshrc' or reopen your terminal to apply changes"

# Check if zsh is installed
check-zsh:
	@echo "Checking zsh..."
	@if command -v zsh >/dev/null 2>&1; then \
		echo "zsh is installed: $$(zsh --version)"; \
	else \
		echo "zsh is not installed. Installing..."; \
		sudo apt install -y zsh; \
		echo "zsh installation complete"; \
	fi

# Install Oh My Zsh
install-omz:
	@echo "Checking Oh My Zsh..."
	@if [ -d "$$HOME/.oh-my-zsh" ]; then \
		echo "Oh My Zsh is already installed"; \
	else \
		echo "Installing Oh My Zsh..."; \
		sh -c "$$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended || true; \
		echo "Oh My Zsh installation complete"; \
	fi

# Install Homebrew (Linux version)
install-brew:
	@echo "Checking Homebrew..."
	@if command -v brew >/dev/null 2>&1; then \
		echo "Homebrew is already installed: $$(brew --version | head -n 1)"; \
	else \
		echo "Installing Homebrew for Linux..."; \
		/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		echo "Homebrew installation complete"; \
	fi

# Install zsh plugins
install-plugins:
	@echo "Installing zsh plugins..."
	@ZSH_CUSTOM=$${ZSH_CUSTOM:-$$HOME/.oh-my-zsh/custom}; \
	\
	if [ -d "$$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]; then \
		echo "zsh-syntax-highlighting is already installed"; \
	else \
		echo "Cloning zsh-syntax-highlighting..."; \
		git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $$ZSH_CUSTOM/plugins/zsh-syntax-highlighting; \
		echo "zsh-syntax-highlighting installation complete"; \
	fi; \
	\
	if [ -d "$$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]; then \
		echo "zsh-autosuggestions is already installed"; \
	else \
		echo "Cloning zsh-autosuggestions..."; \
		git clone https://github.com/zsh-users/zsh-autosuggestions $$ZSH_CUSTOM/plugins/zsh-autosuggestions; \
		echo "zsh-autosuggestions installation complete"; \
	fi

# Setup .zshrc configuration file
setup-zshrc:
	@echo "Setting up .zshrc configuration..."
	@if [ -f "$$HOME/.zshrc" ]; then \
		echo "Backing up existing ~/.zshrc to ~/.zshrc.backup.$$(date +%Y%m%d_%H%M%S)"; \
		cp $$HOME/.zshrc $$HOME/.zshrc.backup.$$(date +%Y%m%d_%H%M%S); \
	fi
	@cp .zshrc $$HOME/.zshrc
	@echo ".zshrc configuration updated"

# Setup SSH key permissions
setup-ssh:
	@echo "Setting up SSH key permissions..."
	@if [ -f "$$HOME/.ssh/id_rsa" ]; then \
		chmod 600 $$HOME/.ssh/id_rsa; \
		echo "SSH key permissions set to 600"; \
	else \
		echo "~/.ssh/id_rsa not found, skipping"; \
	fi

# Create conda environment
setup-conda:
	@echo "Setting up Miniconda environment..."
	@if command -v conda >/dev/null 2>&1; then \
		if conda env list | grep -q "^sglang-jax "; then \
			echo "conda environment 'sglang-jax' already exists"; \
		else \
			echo "Creating conda environment 'sglang-jax'..."; \
			conda create --name sglang-jax python=3.12 -c conda-forge -y; \
			echo "conda environment creation complete"; \
			echo "Activate with: conda activate sglang-jax"; \
		fi \
	else \
		echo "conda is not installed, skipping environment creation"; \
		echo "Please install Miniconda or Anaconda first"; \
	fi

