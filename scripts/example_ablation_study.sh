#!/bin/bash
# Example bash script for running ablation study
# This script demonstrates how to run the ablation study with different configurations

# Activate virtual environment (adjust path as needed)
# source .venv-1/bin/activate

echo -e "\033[1;36mGTNet Ablation Study Examples\033[0m"
echo -e "\033[1;36m==============================\033[0m"
echo ""

# Example 1: Quick smoke test
echo -e "\033[1;33mExample 1: Quick Smoke Test (5 epochs, 200 samples)\033[0m"
echo -e "\033[1;32mCommand:\033[0m"
echo "python scripts/run_ablation_study.py --data-dir data/processed/multi_agent --out-dir ablation_results/quick_test --quick-ablation"
echo ""

# Example 2: Full ablation study on single town
echo -e "\033[1;33mExample 2: Full Ablation Study (30 epochs, all samples)\033[0m"
echo -e "\033[1;32mCommand:\033[0m"
echo "python scripts/run_ablation_study.py --data-dir data/processed/town01 --out-dir ablation_results/town01 --epochs 30 --batch-size 16 --seed 42"
echo ""

# Example 3: Ablation study with multiple towns
echo -e "\033[1;33mExample 3: Multi-Town Ablation Study\033[0m"
echo -e "\033[1;32mCommand:\033[0m"
echo "python scripts/run_ablation_study.py --data-dir data/processed/town01 data/processed/town02 data/processed/town03 --out-dir ablation_results/multi_town --epochs 30"
echo ""

# Example 4: GPU training with larger batch size
echo -e "\033[1;33mExample 4: GPU Training with Larger Batch\033[0m"
echo -e "\033[1;32mCommand:\033[0m"
echo "python scripts/run_ablation_study.py --data-dir data/processed/multi_agent --out-dir ablation_results/gpu --device cuda --batch-size 32 --epochs 30"
echo ""

# Example 5: Limited samples for testing
echo -e "\033[1;33mExample 5: Limited Samples Test\033[0m"
echo -e "\033[1;32mCommand:\033[0m"
echo "python scripts/run_ablation_study.py --data-dir data/processed/multi_agent --out-dir ablation_results/limited --limit-samples 500 --epochs 10"
echo ""

# Prompt user to run an example
echo -e "\033[1;36mSelect an example to run (1-5), or press Enter to exit:\033[0m"
read -r choice

case $choice in
    1)
        echo -e "\033[1;32mRunning Example 1: Quick Smoke Test...\033[0m"
        python scripts/run_ablation_study.py \
            --data-dir data/processed/multi_agent \
            --out-dir ablation_results/quick_test \
            --quick-ablation
        ;;
    2)
        echo -e "\033[1;32mRunning Example 2: Full Ablation Study...\033[0m"
        python scripts/run_ablation_study.py \
            --data-dir data/processed/town01 \
            --out-dir ablation_results/town01 \
            --epochs 30 \
            --batch-size 16 \
            --seed 42
        ;;
    3)
        echo -e "\033[1;32mRunning Example 3: Multi-Town Ablation Study...\033[0m"
        python scripts/run_ablation_study.py \
            --data-dir data/processed/town01 data/processed/town02 data/processed/town03 \
            --out-dir ablation_results/multi_town \
            --epochs 30
        ;;
    4)
        echo -e "\033[1;32mRunning Example 4: GPU Training...\033[0m"
        python scripts/run_ablation_study.py \
            --data-dir data/processed/multi_agent \
            --out-dir ablation_results/gpu \
            --device cuda \
            --batch-size 32 \
            --epochs 30
        ;;
    5)
        echo -e "\033[1;32mRunning Example 5: Limited Samples Test...\033[0m"
        python scripts/run_ablation_study.py \
            --data-dir data/processed/multi_agent \
            --out-dir ablation_results/limited \
            --limit-samples 500 \
            --epochs 10
        ;;
    *)
        echo -e "\033[1;33mExiting...\033[0m"
        ;;
esac

echo ""
echo -e "\033[1;36mDone!\033[0m"
