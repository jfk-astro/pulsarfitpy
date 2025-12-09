# Build script for Windows

set -e

BIN_DIR="bin"
OUTPUT="$BIN_DIR/pulsar-cli"
MAIN_PATH="./cmd/pulsar-cli"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'

show_help() {
    cat << EOF
pulsarfitpy Build Script

Usage:
  ./build.sh              Build the TUI application
  ./build.sh run          Build and run the application
  ./build.sh clean        Clean build artifacts
  ./build.sh help         Show this help message

Examples:
  ./build.sh              # Just build
  ./build.sh run          # Build and run
  ./build.sh clean        # Clean bin directory

EOF
}

clean_build() {
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"

    if [ -d "$BIN_DIR" ]; then
        rm -rf "$BIN_DIR"
        echo -e "${GREEN}✓ Cleaned $BIN_DIR${NC}"
    else
        echo -e "${GRAY}Nothing to clean${NC}"
    fi
}

build_app() {
    echo -e "${CYAN}Building pulsarfitpy TUI...${NC}"

    if [ ! -d "$BIN_DIR" ]; then
        mkdir -p "$BIN_DIR"
    fi
    
    echo -e "${GRAY}Checking dependencies...${NC}"
    go mod download
    
    echo -e "${GRAY}Compiling...${NC}"
    go build -o "$OUTPUT" "$MAIN_PATH"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful: $OUTPUT${NC}"
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            size=$(stat -f%z "$OUTPUT" | awk '{print $1/1024/1024}')
        else
            size=$(stat -c%s "$OUTPUT" | awk '{print $1/1024/1024}')
        fi
        
        echo -e "${GRAY}Binary size: $(printf "%.2f" $size) MB${NC}"
        
        chmod +x "$OUTPUT"
        
        return 0
    else
        echo -e "${RED}✗ Build failed${NC}"
        return 1
    fi
}

run_app() {
    if [ -f "$OUTPUT" ]; then
        echo -e "\n${CYAN}Starting application...${NC}"
        echo -e "${GRAY}----------------------------------------${NC}"
        "$OUTPUT"
    else
        echo -e "${RED}✗ Executable not found: $OUTPUT${NC}"
        echo -e "${YELLOW}Run without 'run' argument to build first${NC}"
        exit 1
    fi
}

case "${1:-build}" in
    help|--help|-h)
        show_help
        exit 0
        ;;
    clean)
        clean_build
        exit 0
        ;;
    run)
        if build_app; then
            run_app
        else
            exit 1
        fi
        ;;
    build|"")
        build_app
        exit $?
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Run './build.sh help' for usage information"
        exit 1
        ;;
esac
