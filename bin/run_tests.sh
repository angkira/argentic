#!/bin/bash

# --- Common Setup ---
echo "Ensuring test dependencies are installed..."
uv sync --all-extras
echo "Dependencies checked/installed."

PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
echo "PYTHONPATH configured: $PYTHONPATH"
# --- End Common Setup ---

# ANSI Color Codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m' # Bold Yellow for section titles
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure subordinate scripts are executable
for script_path in ./bin/run_e2e_tests.sh ./bin/run_unit_tests.sh ./bin/run_integration_tests.sh; do
    if [ -f "$script_path" ] && [ ! -x "$script_path" ]; then
        echo "Making $script_path executable..."
        chmod +x "$script_path"
    fi
done

# Function to parse pytest summary output
parse_pytest_summary() {
    local output="$1"
    local details_string=""

    # Try to find the main summary line, typically enclosed in '==='
    # Regex looks for lines like: === ... [number] <status_keyword> ... in ...s ===
    local summary_line=$(echo "$output" | grep -E '^={5,}.*\b(passed|failed|skipped|selected|errors|warnings|xfailed|xpassed)\b.*in [0-9\.]+s.*={5,}$')
    
    # Fallback if the above doesn't match (e.g., simpler summary or slight variations)
    if [ -z "$summary_line" ]; then
        summary_line=$(echo "$output" | grep -E '^={5,}.*[0-9]+ (passed|failed|skipped|selected|errors|warnings|xfailed|xpassed).*={5,}$')
    fi
    # Additional fallback for lines that might not have "in X.Xs"
    if [ -z "$summary_line" ]; then
        summary_line=$(echo "$output" | grep -E '^={5,}.*\b(passed|failed|skipped|selected|errors|warnings|xfailed|xpassed)\b.*={5,}$' | tail -n 1)
    fi

    if [ -n "$summary_line" ]; then
        local p_c=$(echo "$summary_line" | grep -o '[0-9]* passed' | awk '{print $1+0}')
        local f_c=$(echo "$summary_line" | grep -o '[0-9]* failed' | awk '{print $1+0}')
        local s_c=$(echo "$summary_line" | grep -o '[0-9]* skipped' | awk '{print $1+0}')
        local d_c=$(echo "$summary_line" | grep -o '[0-9]* deselected' | awk '{print $1+0}')
        local e_c=$(echo "$summary_line" | grep -o '[0-9]* errors' | awk '{print $1+0}')
        local w_c=$(echo "$summary_line" | grep -o '[0-9]* warnings' | awk '{print $1+0}')
        local xf_c=$(echo "$summary_line" | grep -o '[0-9]* xfailed' | awk '{print $1+0}')
        local xp_c=$(echo "$summary_line" | grep -o '[0-9]* xpassed' | awk '{print $1+0}')

        details_string=" (P:${p_c:-0}"
        if [ "${f_c:-0}" -gt 0 ]; then details_string+=", ${RED}F:${f_c}${NC}"; else details_string+=", F:0"; fi
        if [ "${e_c:-0}" -gt 0 ]; then details_string+=", ${RED}E:${e_c}${NC}"; fi # Errors are critical
        if [ "${s_c:-0}" -gt 0 ]; then details_string+=", S:${s_c}"; fi
        if [ "${d_c:-0}" -gt 0 ]; then details_string+=", D:${d_c}"; fi
        if [ "${xf_c:-0}" -gt 0 ]; then details_string+=", XF:${xf_c}"; fi
        if [ "${xp_c:-0}" -gt 0 ]; then details_string+=", XP:${xp_c}"; fi
        if [ "${w_c:-0}" -gt 0 ]; then details_string+=", W:${w_c}"; fi
        details_string+=")"
    else
        details_string=" ${YELLOW}(Summary not parsed)${NC}"
    fi
    echo "$details_string"
}


# Behavior based on arguments
if [ $# -eq 0 ]; then
    # --- NO ARGUMENTS: Run all test suites ---
    echo "No arguments provided. Defaulting to run all test suites: E2E (with Docker), then Unit, then Integration."
    
    FINAL_EXIT_CODE=0
    E2E_STATUS="${YELLOW}PENDING${NC}"
    UNIT_STATUS="${YELLOW}PENDING${NC}"
    INTEGRATION_STATUS="${YELLOW}PENDING${NC}"
    PASSED_SUITES=0
    TOTAL_SUITES=3

    # 1. Run E2E tests
    echo -e "\n${BLUE}------------------------------------${NC}"
    echo -e "${BLUE} RUNNING E2E TESTS (with Docker)  ${NC}"
    echo -e "${BLUE}------------------------------------${NC}"
    E2E_OUTPUT_FILE=$(mktemp)
    ./bin/run_e2e_tests.sh --start-docker --force-restart-docker --stop-docker > >(tee "${E2E_OUTPUT_FILE}") 2>&1
    E2E_EXIT_CODE=$?
    E2E_OUTPUT=$(cat "${E2E_OUTPUT_FILE}")
    rm "${E2E_OUTPUT_FILE}"
    E2E_DETAILS=$(parse_pytest_summary "${E2E_OUTPUT}")
    if [ $E2E_EXIT_CODE -ne 0 ]; then
        E2E_STATUS="${RED}FAILED (Exit Code: $E2E_EXIT_CODE)${NC}${E2E_DETAILS}"
        FINAL_EXIT_CODE=$E2E_EXIT_CODE
    else
        E2E_STATUS="${GREEN}PASSED${NC}${E2E_DETAILS}"
        PASSED_SUITES=$((PASSED_SUITES + 1))
    fi

    # 2. Run Unit tests
    echo -e "\n${BLUE}------------------------------------${NC}"
    echo -e "${BLUE}        RUNNING UNIT TESTS        ${NC}"
    echo -e "${BLUE}------------------------------------${NC}"
    UNIT_OUTPUT_FILE=$(mktemp)
    ./bin/run_unit_tests.sh > >(tee "${UNIT_OUTPUT_FILE}") 2>&1
    UNIT_EXIT_CODE=$?
    UNIT_OUTPUT=$(cat "${UNIT_OUTPUT_FILE}")
    rm "${UNIT_OUTPUT_FILE}"
    UNIT_DETAILS=$(parse_pytest_summary "${UNIT_OUTPUT}")
    if [ $UNIT_EXIT_CODE -ne 0 ]; then
        UNIT_STATUS="${RED}FAILED (Exit Code: $UNIT_EXIT_CODE)${NC}${UNIT_DETAILS}"
        if [ $FINAL_EXIT_CODE -eq 0 ]; then FINAL_EXIT_CODE=$UNIT_EXIT_CODE; fi
    else
        UNIT_STATUS="${GREEN}PASSED${NC}${UNIT_DETAILS}"
        PASSED_SUITES=$((PASSED_SUITES + 1))
    fi

    # 3. Run Integration tests
    echo -e "\n${BLUE}------------------------------------${NC}"
    echo -e "${BLUE}    RUNNING INTEGRATION TESTS     ${NC}"
    echo -e "${BLUE}------------------------------------${NC}"
    INTEGRATION_OUTPUT_FILE=$(mktemp)
    ./bin/run_integration_tests.sh > >(tee "${INTEGRATION_OUTPUT_FILE}") 2>&1
    INTEGRATION_EXIT_CODE=$?
    INTEGRATION_OUTPUT=$(cat "${INTEGRATION_OUTPUT_FILE}")
    rm "${INTEGRATION_OUTPUT_FILE}"
    INTEGRATION_DETAILS=$(parse_pytest_summary "${INTEGRATION_OUTPUT}")
    if [ $INTEGRATION_EXIT_CODE -ne 0 ]; then
        INTEGRATION_STATUS="${RED}FAILED (Exit Code: $INTEGRATION_EXIT_CODE)${NC}${INTEGRATION_DETAILS}"
        if [ $FINAL_EXIT_CODE -eq 0 ]; then FINAL_EXIT_CODE=$INTEGRATION_EXIT_CODE; fi
    else
        INTEGRATION_STATUS="${GREEN}PASSED${NC}${INTEGRATION_DETAILS}"
        PASSED_SUITES=$((PASSED_SUITES + 1))
    fi
    
    # Final Summary Section
    echo -e "\n${YELLOW}------------------------------------${NC}"
    echo -e "${YELLOW}          OVERALL RESULT          ${NC}"
    echo -e "${YELLOW}------------------------------------${NC}"
    echo -e "E2E Tests:           $E2E_STATUS"
    echo -e "Unit Tests:          $UNIT_STATUS"
    echo -e "Integration Tests:   $INTEGRATION_STATUS"
    echo -e "${YELLOW}------------------------------------${NC}"
    
    SUMMARY_COLOR=$GREEN
    if [ $PASSED_SUITES -ne $TOTAL_SUITES ]; then
        SUMMARY_COLOR=$RED
    fi
    echo -e "${SUMMARY_COLOR}Summary: $PASSED_SUITES / $TOTAL_SUITES test suites passed.${NC}"

    if [ $FINAL_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}One or more test suites FAILED. Overall script exit code: $FINAL_EXIT_CODE${NC}"
    else
        echo -e "${GREEN}All test suites PASSED successfully!${NC}"
    fi
    exit $FINAL_EXIT_CODE

else
    # --- ARGUMENTS PROVIDED: Dispatch or run pytest ---
    RUN_E2E_DISPATCH_SCRIPT=false
    for arg in "$@"; do
        if [[ "$arg" == *"tests/core/messager/e2e"* || \
              "$arg" == *"-m e2e"* || \
              "$arg" == "--start-docker" || \
              "$arg" == "--force-restart-docker" || \
              "$arg" == "--stop-docker" || \
              "$arg" == "--docker-only" ]]; then
            RUN_E2E_DISPATCH_SCRIPT=true
            break
        fi
    done

    if [ "$RUN_E2E_DISPATCH_SCRIPT" = true ]; then
        echo "E2E context or Docker flags detected in arguments. Forwarding all arguments to bin/run_e2e_tests.sh..."
        # For dispatched calls, live output is handled by the sub-script directly.
        # No special teeing needed here, as we are not parsing its summary in this branch.
        ./bin/run_e2e_tests.sh "$@"
    else
        echo "Arguments provided, and not an E2E context/no Docker flags detected. Running pytest directly..."
        # Pytest will output live to terminal directly.
        python -m pytest "$@"
    fi
    exit $?
fi 