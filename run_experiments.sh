#!/bin/bash

# Chinese-Thai Medical Translation Experiment Runner
# สคริปต์สำหรับรันการทดลองแปลภาษาทางการแพทย์จีน-ไทย

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python installation
check_python() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    print_info "Using Python: $PYTHON_CMD"
    $PYTHON_CMD --version
}

# Function to check and install requirements
setup_environment() {
    print_info "Setting up environment..."
    
    # Check if pip is available
    if ! command_exists pip && ! command_exists pip3; then
        print_error "pip is not installed"
        exit 1
    fi
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_info "Installing requirements..."
        $PYTHON_CMD -m pip install -r requirements.txt
        print_success "Requirements installed successfully"
    else
        print_warning "requirements.txt not found"
    fi
    
    # Create necessary directories
    mkdir -p data output models logs
    print_info "Created necessary directories"
}

# Function to check GPU availability
check_gpu() {
    print_info "Checking GPU availability..."
    $PYTHON_CMD -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Running on CPU')
"
}

# Function to create sample data
create_sample_data() {
    print_info "Creating sample data for testing..."
    $PYTHON_CMD run_experiment.py --create_sample_data
    print_success "Sample data created"
}

# Function to run data analysis
run_data_analysis() {
    print_info "Running data analysis..."
    $PYTHON_CMD run_experiment.py --run_analysis
    print_success "Data analysis completed"
}

# Function to run quick test
run_quick_test() {
    print_info "Running quick test with OPUS model..."
    $PYTHON_CMD run_experiment.py \
        --experiment_name "quick_test_$(date +%y%m%d_%H%M%S)" \
        --model_config opus \
        --eval_splits dev \
        --create_sample_data
    print_success "Quick test completed"
}

# Function to run full experiment
run_full_experiment() {
    local experiment_name=${1:-"full_experiment_$(date +%y%m%d_%H%M%S)"}
    print_info "Running full experiment: $experiment_name"
    
    $PYTHON_CMD run_experiment.py \
        --experiment_name "$experiment_name" \
        --model_config all \
        --eval_splits dev test
    
    print_success "Full experiment completed: $experiment_name"
}

# Function to run specific model
run_model_experiment() {
    local model_type=${1:-opus}
    local experiment_name="model_${model_type}_$(date +%y%m%d_%H%M%S)"
    
    print_info "Running experiment with $model_type model..."
    
    $PYTHON_CMD run_experiment.py \
        --experiment_name "$experiment_name" \
        --model_config "$model_type" \
        --eval_splits dev
    
    print_success "Model experiment completed: $experiment_name"
}

# Function to generate submission
generate_submission() {
    local model_type=${1:-opus}
    local experiment_name="submission_${model_type}_$(date +%y%m%d_%H%M%S)"
    
    print_info "Generating submission with $model_type model..."
    
    $PYTHON_CMD main.py \
        --mode predict \
        --model_name "Helsinki-NLP/opus-mt-zh-en" \
        --output_dir "output/$experiment_name"
    
    print_success "Submission generated: output/$experiment_name/submission.txt"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                     Setup environment and install requirements"
    echo "  check                     Check system requirements and GPU"
    echo "  sample-data               Create sample data for testing"
    echo "  analyze                   Run data analysis"
    echo "  quick-test               Run quick test with sample data"
    echo "  full-experiment [name]    Run full experiment with all models"
    echo "  model-experiment <type>   Run experiment with specific model (opus|nllb|mt5)"
    echo "  submission <type>         Generate submission file with specific model"
    echo "  help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                          # Setup environment"
    echo "  $0 quick-test                     # Quick test"
    echo "  $0 full-experiment my_exp         # Full experiment named 'my_exp'"
    echo "  $0 model-experiment opus          # Test OPUS model only"
    echo "  $0 submission nllb                # Generate submission with NLLB model"
}

# Function to run system check
run_system_check() {
    print_info "Running system check..."
    
    # Check Python
    check_python
    
    # Check GPU
    check_gpu
    
    # Check disk space
    print_info "Checking disk space..."
    df -h .
    
    # Check memory
    print_info "Checking memory..."
    if command_exists free; then
        free -h
    elif command_exists vm_stat; then
        vm_stat
    fi
    
    print_success "System check completed"
}

# Main execution
main() {
    local command=${1:-help}
    
    echo "=================================================="
    echo "Chinese-Thai Medical Translation Experiment Runner"
    echo "=================================================="
    echo ""
    
    case $command in
        setup)
            check_python
            setup_environment
            ;;
        check)
            run_system_check
            ;;
        sample-data)
            check_python
            create_sample_data
            ;;
        analyze)
            check_python
            run_data_analysis
            ;;
        quick-test)
            check_python
            run_quick_test
            ;;
        full-experiment)
            check_python
            run_full_experiment "$2"
            ;;
        model-experiment)
            check_python
            run_model_experiment "$2"
            ;;
        submission)
            check_python
            generate_submission "$2"
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
