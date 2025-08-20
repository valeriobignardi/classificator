#!/bin/bash

# 🚀 Universal Deploy Installer with Guided Wizard
# Version: 2.0.0
# Installs deployment capabilities to any project

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

INSTALL_DIR="deploy-tools"
TEMPLATES_DIR="$(dirname "$0")/templates"
INSTALLER_VERSION="2.0.0"

# Configuration file for storing values
CONFIG_FILE="/tmp/deploy_config_$$"

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}🔥 $1${NC}"
}

# Set configuration value
set_config() {
    local key="$1"
    local value="$2"
    echo "$key=$value" >> "$CONFIG_FILE"
}

# Get configuration value
get_config() {
    local key="$1"
    grep "^$key=" "$CONFIG_FILE" 2>/dev/null | cut -d'=' -f2- | tail -1
}

# Initialize config file
init_config() {
    > "$CONFIG_FILE"
}

# Clean up config file
cleanup_config() {
    rm -f "$CONFIG_FILE"
}

# Detect project type
detect_project_type() {
    if [ -f "package.json" ]; then
        echo "nodejs"
    elif [ -f "requirements.txt" ] || [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
        echo "python"
    elif [ -f "go.mod" ]; then
        echo "go"
    elif [ -f "pom.xml" ] || [ -f "build.gradle" ]; then
        echo "java"
    elif [ -f "docker-compose.yml" ]; then
        echo "docker"
    else
        echo "generic"
    fi
}

# Extract placeholders from templates
extract_placeholders() {
    local template_file="$1"
    grep -oE '\{\{[A-Z_]+\}\}' "$template_file" 2>/dev/null | sed 's/[{}]//g' | sort -u || true
}

# Get all unique placeholders from all templates
get_all_placeholders() {
    local all_placeholders=""
    
    for template_file in "$TEMPLATES_DIR"/*.template.*; do
        if [ -f "$template_file" ]; then
            local placeholders=$(extract_placeholders "$template_file")
            all_placeholders="$all_placeholders $placeholders"
        fi
    done
    
    echo "$all_placeholders" | tr ' ' '\n' | sort -u | grep -v '^$'
}

# Get default value for placeholder
get_default_value() {
    local placeholder="$1"
    local project_type="$2"
    local project_name="$3"
    
    case $placeholder in
        "PROJECT_NAME") echo "$project_name" ;;
        "PROJECT_TYPE") echo "$project_type" ;;
        "MAIN_PORT") 
            case $project_type in
                "nodejs") echo "3000" ;;
                "python") echo "8000" ;;
                "go") echo "8080" ;;
                "java") echo "8080" ;;
                *) echo "3000" ;;
            esac ;;
        "API_PORT") echo "$(get_config MAIN_PORT || echo "3001")" ;;
        "WS_PORT") echo "8080" ;;
        "DATABASE_PORT") echo "5432" ;;
        "REDIS_PORT") echo "6379" ;;
        "ENVIRONMENT") echo "production" ;;
        "DATABASE_TYPE") echo "postgres" ;;
        "DATABASE_IMAGE") echo "postgres:15-alpine" ;;
        "DATABASE_NAME") echo "${project_name}_db" ;;
        "DATABASE_USER") echo "postgres" ;;
        "DATABASE_PASSWORD") echo "$(openssl rand -base64 12 2>/dev/null || echo "changeme123")" ;;
        "BASE_IMAGE") 
            case $project_type in
                "nodejs") echo "node:18-alpine" ;;
                "python") echo "python:3.11-alpine" ;;
                "go") echo "golang:1.21-alpine" ;;
                "java") echo "openjdk:17-alpine" ;;
                *) echo "alpine:latest" ;;
            esac ;;
        "BUILD_COMMAND") 
            case $project_type in
                "nodejs") echo "npm run build" ;;
                "python") echo "pip install -r requirements.txt" ;;
                "docker") echo "docker-compose build" ;;
                *) echo "echo 'No build needed'" ;;
            esac ;;
        "START_COMMAND") 
            case $project_type in
                "nodejs") echo "[\"npm\", \"start\"]" ;;
                "python") echo "[\"python\", \"app.py\"]" ;;
                "docker") echo "docker-compose up -d" ;;
                *) echo "[\"echo\", \"Hello World\"]" ;;
            esac ;;
        "INSTALL_COMMAND") 
            case $project_type in
                "nodejs") echo "RUN npm ci --only=production" ;;
                "python") echo "RUN pip install --no-cache-dir -r requirements.txt" ;;
                *) echo "RUN echo 'No install needed'" ;;
            esac ;;
        "COPY_PACKAGE_FILES") 
            case $project_type in
                "nodejs") echo "COPY package*.json ./" ;;
                "python") echo "COPY requirements.txt ./" ;;
                *) echo "COPY . ./" ;;
            esac ;;
        "HEALTH_ENDPOINT") echo "/health" ;;
        "SERVER_HOST") echo "your-server-ip" ;;
        "SSH_USER") echo "ubuntu" ;;
        "SSH_KEY") echo "~/.ssh/id_rsa" ;;
        "REMOTE_PROJECT_DIR") echo "/home/ubuntu/$project_name" ;;
        "DOMAIN") echo "" ;;
        "DATABASE_URL") echo "postgresql://$(get_config DATABASE_USER || echo "postgres"):$(get_config DATABASE_PASSWORD || echo "changeme123")@database:5432/$(get_config DATABASE_NAME || echo "${project_name}_db")" ;;
        "BACKUP_BEFORE_DEPLOY") echo "true" ;;
        "RUN_HEALTH_CHECK") echo "true" ;;
        "SSL_ENABLED") echo "false" ;;
        "CREATED") echo "$(date)" ;;
        "INSTALLER_VERSION") echo "$INSTALLER_VERSION" ;;
        *) echo "" ;;
    esac
}

# Get user-friendly description for placeholder
get_placeholder_description() {
    local placeholder="$1"
    
    case $placeholder in
        "PROJECT_NAME") echo "Project name (used for containers, directories)" ;;
        "PROJECT_TYPE") echo "Project type (nodejs, python, docker, etc.)" ;;
        "MAIN_PORT") echo "Main application port" ;;
        "API_PORT") echo "API server port (if different from main)" ;;
        "WS_PORT") echo "WebSocket server port" ;;
        "DATABASE_PORT") echo "Database port" ;;
        "REDIS_PORT") echo "Redis port" ;;
        "ENVIRONMENT") echo "Deployment environment" ;;
        "DATABASE_TYPE") echo "Database type" ;;
        "DATABASE_IMAGE") echo "Database Docker image" ;;
        "DATABASE_NAME") echo "Database name" ;;
        "DATABASE_USER") echo "Database username" ;;
        "DATABASE_PASSWORD") echo "Database password" ;;
        "BASE_IMAGE") echo "Docker base image" ;;
        "BUILD_COMMAND") echo "Build command" ;;
        "START_COMMAND") echo "Container start command (JSON array format)" ;;
        "INSTALL_COMMAND") echo "Dependencies install command" ;;
        "COPY_PACKAGE_FILES") echo "Package files to copy" ;;
        "HEALTH_ENDPOINT") echo "Health check endpoint" ;;
        "SERVER_HOST") echo "Server IP or hostname" ;;
        "SSH_USER") echo "SSH username" ;;
        "SSH_KEY") echo "SSH private key path" ;;
        "REMOTE_PROJECT_DIR") echo "Project directory on server" ;;
        "DOMAIN") echo "Domain name (optional, for SSL)" ;;
        "DATABASE_URL") echo "Complete database connection URL" ;;
        "BACKUP_BEFORE_DEPLOY") echo "Backup before deployment (true/false)" ;;
        "RUN_HEALTH_CHECK") echo "Run health check after deployment (true/false)" ;;
        "SSL_ENABLED") echo "Enable SSL certificate setup (true/false)" ;;
        *) echo "$placeholder" ;;
    esac
}

# Interactive wizard to collect configuration
run_configuration_wizard() {
    local project_name=$(basename "$PWD")
    local project_type=$(detect_project_type)
    
    log_header "🚀 Universal Deploy Configuration Wizard"
    echo ""
    log_info "Detected project type: $project_type"
    log_info "Project directory: $PWD"
    echo ""
    
    log_info "📋 Scanning templates for configuration options..."
    local placeholders=$(get_all_placeholders)
    
    if [ -z "$placeholders" ]; then
        log_error "No templates found in $TEMPLATES_DIR"
        exit 1
    fi
    
    echo ""
    log_header "🔧 Please provide the following configuration:"
    echo ""
    
    # Collect configuration values
    for placeholder in $placeholders; do
        local description=$(get_placeholder_description "$placeholder")
        local default_value=$(get_default_value "$placeholder" "$project_type" "$project_name")
        
        # Skip auto-generated fields
        case $placeholder in
            "CREATED"|"INSTALLER_VERSION"|"DATABASE_URL") continue ;;
        esac
        
        echo -e "${BLUE}$placeholder${NC} - $description"
        if [ -n "$default_value" ]; then
            read -p "  Value [$default_value]: " user_value
            CONFIG_VALUES[$placeholder]="${user_value:-$default_value}"
        else
            read -p "  Value: " user_value
            CONFIG_VALUES[$placeholder]="$user_value"
        fi
        echo ""
    done
    
    # Generate derived values
    CONFIG_VALUES["CREATED"]=$(date)
    CONFIG_VALUES["INSTALLER_VERSION"]="$INSTALLER_VERSION"
    CONFIG_VALUES["DATABASE_URL"]="postgresql://${CONFIG_VALUES[DATABASE_USER]}:${CONFIG_VALUES[DATABASE_PASSWORD]}@database:5432/${CONFIG_VALUES[DATABASE_NAME]}"
    
    echo ""
    log_success "Configuration completed!"
    echo ""
    
    # Show summary
    log_header "📋 Configuration Summary:"
    for placeholder in $placeholders; do
        echo "  $placeholder = ${CONFIG_VALUES[$placeholder]}"
    done
    echo ""
    
    read -p "Proceed with installation? [Y/n]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "Installation cancelled."
        exit 0
    fi
}

# Process template file and replace placeholders
process_template() {
    local template_file="$1"
    local output_file="$2"
    
    log_info "Processing template: $(basename "$template_file")"
    
    local content=$(cat "$template_file")
    
    # Replace all placeholders
    for placeholder in "${!CONFIG_VALUES[@]}"; do
        local value="${CONFIG_VALUES[$placeholder]}"
        content=$(echo "$content" | sed "s|{{$placeholder}}|$value|g")
    done
    
    echo "$content" > "$output_file"
    log_success "Generated: $(basename "$output_file")"
}

install_local() {
    log_info "Installing Universal Deploy Tools with guided setup..."
    
    # Check if templates directory exists
    if [ ! -d "$TEMPLATES_DIR" ]; then
        log_error "Templates directory not found: $TEMPLATES_DIR"
        log_info "Make sure you're running this script from the universal-deploy-standalone directory"
        exit 1
    fi
    
    # Run configuration wizard
    run_configuration_wizard
    
    if [ -d "$INSTALL_DIR" ]; then
        log_warning "Deploy tools already installed. Updating..."
        rm -rf "$INSTALL_DIR"
    fi
    
    mkdir -p "$INSTALL_DIR"
    
    # Copy universal deploy script
    if [ -f "universal-deploy.sh" ]; then
        cp "universal-deploy.sh" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/universal-deploy.sh"
    else
        log_error "universal-deploy.sh not found!"
        exit 1
    fi
    
    log_info "🔄 Processing templates..."
    echo ""
    
    # Process all template files
    for template_file in "$TEMPLATES_DIR"/*.template.*; do
        if [ -f "$template_file" ]; then
            local filename=$(basename "$template_file")
            local output_name="${filename/.template/}"
            local output_file="$INSTALL_DIR/$output_name"
            
            process_template "$template_file" "$output_file"
        fi
    done
    
    # Create main deployment config
    if [ -f "$INSTALL_DIR/deploy-config.env" ]; then
        cp "$INSTALL_DIR/deploy-config.env" "deploy-config.env"
        log_success "Created: deploy-config.env"
    fi
    
    # Create wrapper script
    cat > "deploy" << 'EOF'
#!/bin/bash
# Universal Deploy Wrapper
exec ./deploy-tools/universal-deploy.sh "$@"
EOF
    chmod +x deploy
    log_success "Created: deploy (wrapper script)"
    
    # Create quick setup script
    cat > "$INSTALL_DIR/quick-setup.sh" << 'EOF'
#!/bin/bash
# Quick setup - regenerates Docker files from templates

set -e

if [ ! -f "deploy-config.env" ]; then
    echo "❌ deploy-config.env not found. Run './deploy config' first."
    exit 1
fi

source deploy-config.env

echo "🚀 Quick Setup for $PROJECT_TYPE project"
echo ""

# Generate Dockerfile if it doesn't exist
if [ ! -f "Dockerfile" ] && [ -f "deploy-tools/Dockerfile" ]; then
    echo "📦 Creating Dockerfile..."
    cp "deploy-tools/Dockerfile" "Dockerfile"
fi

# Generate docker-compose.yml if it doesn't exist
if [ ! -f "docker-compose.yml" ] && [ -f "deploy-tools/docker-compose.yml" ]; then
    echo "🐳 Creating docker-compose.yml..."
    cp "deploy-tools/docker-compose.yml" "docker-compose.yml"
fi

# Generate nginx.conf if needed
if [ -n "$DOMAIN" ] && [ ! -f "nginx.conf" ] && [ -f "deploy-tools/nginx.conf" ]; then
    echo "🌐 Creating nginx.conf..."
    cp "deploy-tools/nginx.conf" "nginx.conf"
fi

echo "✅ Quick setup completed!"
echo "   Run: ./deploy deploy"
EOF
    chmod +x "$INSTALL_DIR/quick-setup.sh"
    log_success "Created: quick-setup.sh"
    
    echo ""
    log_success "🎉 Universal Deploy Tools installed successfully!"
    echo ""
    log_header "📋 What was created:"
    echo "  • deploy-tools/          - Deployment tools and processed templates"
    echo "  • deploy                 - Main deployment command"
    echo "  • deploy-config.env      - Your project configuration"
    echo ""
    log_header "🚀 Next steps:"
    echo "  • ./deploy test          - Test server connection"
    echo "  • ./deploy deploy        - Deploy your project"
    echo "  • ./deploy-tools/quick-setup.sh - Generate Docker files"
    echo ""
    log_header "📁 Generated files ready for deployment:"
    ls -la "$INSTALL_DIR"/ | grep -E '\.(yml|conf|env)$' || echo "  (Files will be shown after processing)"
}

install_from_repo() {
    log_info "Installing from repository..."
    
    if command -v git &> /dev/null; then
        git clone "$REPO_URL" "$INSTALL_DIR"
        chmod +x "$INSTALL_DIR"/*.sh
        ln -sf "$INSTALL_DIR/universal-deploy.sh" deploy
        log_success "Installed from repository!"
    else
        log_warning "Git not found. Installing locally..."
        install_local
    fi
}

create_project_config() {
    log_info "Creating project-specific configuration..."
    
    PROJECT_NAME=$(basename "$PWD")
    
    cat > ".deployrc" << EOF
# Project Deploy Configuration
PROJECT_NAME="$PROJECT_NAME"
CREATED=$(date)

# Override default settings here
# MAIN_PORT=3000
# HEALTH_ENDPOINT="/health"
# BUILD_COMMAND="npm run build"
# START_COMMAND="npm start"
EOF
    
    log_success "Project configuration created in .deployrc"
}

show_usage() {
    echo "🚀 Universal Deploy Installer v$INSTALLER_VERSION"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  install     - Install deploy tools with guided wizard"
    echo "  config      - Create project configuration only"
    echo "  uninstall   - Remove deploy tools"
    echo "  test        - Test template processing"
    echo ""
    echo "After installation, use: ./deploy [command]"
    echo ""
    echo "Templates directory: $TEMPLATES_DIR"
}

test_templates() {
    log_info "Testing template processing..."
    
    if [ ! -d "$TEMPLATES_DIR" ]; then
        log_error "Templates directory not found: $TEMPLATES_DIR"
        exit 1
    fi
    
    log_info "Available templates:"
    for template_file in "$TEMPLATES_DIR"/*.template.*; do
        if [ -f "$template_file" ]; then
            echo "  • $(basename "$template_file")"
            local placeholders=$(extract_placeholders "$template_file")
            if [ -n "$placeholders" ]; then
                echo "    Placeholders: $(echo $placeholders | tr '\n' ' ')"
            fi
        fi
    done
    
    echo ""
    log_info "All unique placeholders found:"
    get_all_placeholders | while read placeholder; do
        if [ -n "$placeholder" ]; then
            echo "  • $placeholder - $(get_placeholder_description "$placeholder")"
        fi
    done
}

uninstall() {
    log_info "Uninstalling deploy tools..."
    
    rm -rf "$INSTALL_DIR"
    rm -f deploy
    rm -f .deployrc
    rm -f deploy-config.env
    rm -f Dockerfile docker-compose.yml nginx.conf
    
    log_success "Universal Deploy Tools uninstalled"
}

case "$1" in
    "install")
        install_local
        ;;
    "config")
        create_project_config
        ;;
    "uninstall")
        uninstall
        ;;
    "test")
        test_templates
        ;;
    *)
        show_usage
        ;;
esac
