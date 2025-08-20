#!/bin/bash

# 🚀 Universal Deploy Script - Deploy any project to EC2/Server
# Version: 1.0.0
# Usage: ./universal-deploy.sh [command]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Default configuration
DEPLOY_CONFIG_FILE="deploy-config.env"
PROJECT_NAME=$(basename "$PWD")
DEFAULT_PROJECT_DIR="/home/ubuntu/$PROJECT_NAME"

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
    elif [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
        echo "docker"
    else
        echo "generic"
    fi
}

# Auto-detect deployment configurations
auto_detect_config() {
    PROJECT_TYPE=$(detect_project_type)
    
    case $PROJECT_TYPE in
        "nodejs")
            MAIN_PORT=3000
            HEALTH_ENDPOINT="/health"
            BUILD_COMMAND="npm install && npm run build"
            START_COMMAND="npm start"
            ;;
        "python")
            MAIN_PORT=8000
            HEALTH_ENDPOINT="/health"
            BUILD_COMMAND="pip install -r requirements.txt"
            START_COMMAND="python main.py"
            ;;
        "docker")
            MAIN_PORT=80
            HEALTH_ENDPOINT="/"
            BUILD_COMMAND="docker-compose build"
            START_COMMAND="docker-compose up -d"
            ;;
        *)
            MAIN_PORT=8080
            HEALTH_ENDPOINT="/"
            BUILD_COMMAND="echo 'No build command configured'"
            START_COMMAND="echo 'No start command configured'"
            ;;
    esac
    
    log_info "Detected project type: $PROJECT_TYPE"
    log_info "Main port: $MAIN_PORT"
}

# Create deployment configuration
create_deploy_config() {
    log_header "Universal Deploy Configuration"
    
    if [ -f "$DEPLOY_CONFIG_FILE" ]; then
        log_info "Existing configuration found."
        read -p "Reconfigure? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    auto_detect_config
    
    echo -e "${BLUE}Server Configuration:${NC}"
    
    # Server details
    read -p "🌐 Server IP/Hostname: " SERVER_HOST
    while [ -z "$SERVER_HOST" ]; do
        log_error "Server IP required!"
        read -p "🌐 Server IP/Hostname: " SERVER_HOST
    done
    
    read -p "👤 SSH User (default: ubuntu): " SSH_USER
    SSH_USER=${SSH_USER:-ubuntu}
    
    read -p "🔑 SSH Key Path (default: ~/.ssh/id_rsa): " SSH_KEY
    SSH_KEY=${SSH_KEY:-~/.ssh/id_rsa}
    SSH_KEY=$(eval echo "$SSH_KEY")
    
    read -p "📁 Project Directory (default: $DEFAULT_PROJECT_DIR): " REMOTE_PROJECT_DIR
    REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR:-$DEFAULT_PROJECT_DIR}
    
    read -p "🌍 Domain (optional): " DOMAIN
    
    # Project-specific configuration
    echo -e "${BLUE}Project Configuration:${NC}"
    
    read -p "🚀 Main Port (default: $MAIN_PORT): " MAIN_PORT_INPUT
    MAIN_PORT=${MAIN_PORT_INPUT:-$MAIN_PORT}
    
    read -p "🏥 Health Endpoint (default: $HEALTH_ENDPOINT): " HEALTH_ENDPOINT_INPUT
    HEALTH_ENDPOINT=${HEALTH_ENDPOINT_INPUT:-$HEALTH_ENDPOINT}
    
    read -p "🔨 Build Command (default: $BUILD_COMMAND): " BUILD_COMMAND_INPUT
    BUILD_COMMAND=${BUILD_COMMAND_INPUT:-$BUILD_COMMAND}
    
    read -p "▶️  Start Command (default: $START_COMMAND): " START_COMMAND_INPUT
    START_COMMAND=${START_COMMAND_INPUT:-$START_COMMAND}
    
    # Environment configuration
    echo -e "${BLUE}Environment Configuration:${NC}"
    read -p "🌍 Environment (default: production): " ENVIRONMENT
    ENVIRONMENT=${ENVIRONMENT:-production}
    
    # Save configuration
    cat > "$DEPLOY_CONFIG_FILE" << EOF
# Universal Deploy Configuration
# Project: $PROJECT_NAME
# Type: $PROJECT_TYPE
# Generated: $(date)

# === SERVER CONFIGURATION ===
SERVER_HOST="$SERVER_HOST"
SSH_USER="$SSH_USER"
SSH_KEY="$SSH_KEY"
REMOTE_PROJECT_DIR="$REMOTE_PROJECT_DIR"
DOMAIN="$DOMAIN"

# === PROJECT CONFIGURATION ===
PROJECT_NAME="$PROJECT_NAME"
PROJECT_TYPE="$PROJECT_TYPE"
MAIN_PORT="$MAIN_PORT"
HEALTH_ENDPOINT="$HEALTH_ENDPOINT"
BUILD_COMMAND="$BUILD_COMMAND"
START_COMMAND="$START_COMMAND"
ENVIRONMENT="$ENVIRONMENT"

# === DEPLOYMENT SETTINGS ===
BACKUP_BEFORE_DEPLOY=true
RESTART_SERVICES=true
RUN_HEALTH_CHECK=true
SETUP_SSL=false
EOF
    
    log_success "Configuration saved to $DEPLOY_CONFIG_FILE"
}

# Load configuration
load_config() {
    if [ ! -f "$DEPLOY_CONFIG_FILE" ]; then
        log_warning "Configuration not found"
        create_deploy_config
    fi
    
    source "$DEPLOY_CONFIG_FILE"
    
    log_info "Configuration loaded:"
    echo "  🌐 Server: $SERVER_HOST"
    echo "  👤 User: $SSH_USER"
    echo "  📁 Directory: $REMOTE_PROJECT_DIR"
    echo "  🚀 Port: $MAIN_PORT"
    echo "  🏗️  Type: $PROJECT_TYPE"
}

# Test SSH connection
test_connection() {
    log_info "Testing SSH connection..."
    
    if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$SSH_USER@$SERVER_HOST" echo "Connection OK" > /dev/null 2>&1; then
        log_success "SSH connection successful"
    else
        log_error "Cannot connect to $SSH_USER@$SERVER_HOST"
        log_info "Check:"
        log_info "1. Server IP: $SERVER_HOST"
        log_info "2. SSH Key: $SSH_KEY"
        log_info "3. Security group/firewall"
        log_info "4. Server status"
        exit 1
    fi
}

# Backup current deployment
backup_deployment() {
    if [ "$BACKUP_BEFORE_DEPLOY" = "true" ]; then
        log_info "Creating backup..."
        
        ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" << EOF
if [ -d "$REMOTE_PROJECT_DIR" ]; then
    BACKUP_DIR="${REMOTE_PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    cp -r "$REMOTE_PROJECT_DIR" "\$BACKUP_DIR"
    echo "✅ Backup created: \$BACKUP_DIR"
else
    echo "ℹ️  No existing deployment to backup"
fi
EOF
    fi
}

# Sync code
sync_code() {
    log_header "Syncing Code"
    
    log_info "Syncing project to $SERVER_HOST..."
    
    # Create exclude patterns based on project type
    EXCLUDE_PATTERNS=""
    case $PROJECT_TYPE in
        "nodejs")
            EXCLUDE_PATTERNS="--exclude 'node_modules' --exclude '.npm' --exclude 'dist' --exclude 'build'"
            ;;
        "python")
            EXCLUDE_PATTERNS="--exclude '__pycache__' --exclude '.venv' --exclude 'venv' --exclude '*.pyc' --exclude '.pytest_cache'"
            ;;
        "docker")
            EXCLUDE_PATTERNS="--exclude 'node_modules' --exclude '__pycache__' --exclude '.venv' --exclude 'venv'"
            ;;
    esac
    
    # Common exclusions
    COMMON_EXCLUDES="--exclude '.git' --exclude '.DS_Store' --exclude '*.log' --exclude '.env' --exclude 'logs/' --exclude 'tmp/' --exclude 'json_scraped/'"
    
    rsync -avz --progress \
        -e "ssh -i $SSH_KEY" \
        $EXCLUDE_PATTERNS \
        $COMMON_EXCLUDES \
        ./ "$SSH_USER@$SERVER_HOST:$REMOTE_PROJECT_DIR/"
    
    log_success "Code synced successfully"
}

# Setup remote environment
setup_remote() {
    log_header "Remote Setup"
    
    log_info "Setting up remote environment..."
    
    ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" << EOF
export TERM=xterm
set -e

cd "$REMOTE_PROJECT_DIR"

echo "🔧 Setting up permissions..."
find . -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

echo "🏗️  Project type: $PROJECT_TYPE"

# Install dependencies based on project type
case "$PROJECT_TYPE" in
    "nodejs")
        if ! command -v node &> /dev/null; then
            echo "📦 Installing Node.js..."
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
        fi
        ;;
    "python")
        if ! command -v python3 &> /dev/null; then
            echo "🐍 Installing Python..."
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
        fi
        ;;
    "docker")
        if ! command -v docker &> /dev/null; then
            echo "🐳 Installing Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker \$USER
            sudo systemctl start docker
            sudo systemctl enable docker
            
            # Install Docker Compose
            sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
        fi
        ;;
esac

echo "✅ Remote setup completed"
EOF
    
    log_success "Remote environment ready"
}

# Deploy application
deploy_app() {
    log_header "Deploying Application"
    
    log_info "Starting deployment..."
    
    ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" << EOF
export TERM=xterm
set -e

cd "$REMOTE_PROJECT_DIR"

echo "🔨 Running build command: $BUILD_COMMAND"
$BUILD_COMMAND

echo "🚀 Starting application: $START_COMMAND"
$START_COMMAND

echo "⏳ Waiting for application to start..."
sleep 10

echo "✅ Deployment completed!"
EOF
    
    log_success "Application deployed"
}

# Health check
health_check() {
    if [ "$RUN_HEALTH_CHECK" = "true" ]; then
        log_header "Health Check"
        
        log_info "Checking application health..."
        
        ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" << EOF
cd "$REMOTE_PROJECT_DIR"

echo "🏥 Testing health endpoint: http://localhost:$MAIN_PORT$HEALTH_ENDPOINT"
if curl -f -s http://localhost:$MAIN_PORT$HEALTH_ENDPOINT > /dev/null; then
    echo "✅ Application is healthy"
else
    echo "❌ Application health check failed"
    exit 1
fi
EOF
    fi
}

# Setup SSL (if domain configured)
setup_ssl() {
    if [ "$SETUP_SSL" = "true" ] && [ -n "$DOMAIN" ]; then
        log_header "SSL Setup"
        
        log_info "Setting up SSL for $DOMAIN..."
        
        ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" << EOF
echo "🔒 Installing Certbot..."
sudo apt-get update
sudo apt-get install -y certbot nginx

echo "🔧 Configuring Nginx..."
sudo tee /etc/nginx/sites-available/$DOMAIN << 'NGINX_CONFIG'
server {
    listen 80;
    server_name $DOMAIN;
    
    location / {
        proxy_pass http://localhost:$MAIN_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINX_CONFIG

sudo ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

echo "🔒 Getting SSL certificate..."
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN || echo "SSL setup failed"
EOF
        
        log_success "SSL configured"
    fi
}

# Show final information
show_final_info() {
    log_header "Deployment Complete!"
    
    echo ""
    echo -e "${GREEN}🎉 $PROJECT_NAME deployed successfully!${NC}"
    echo ""
    echo -e "${BLUE}🌐 Access:${NC}"
    echo "   • Application: http://$SERVER_HOST:$MAIN_PORT"
    
    if [ -n "$DOMAIN" ]; then
        echo "   • Domain:      https://$DOMAIN"
    fi
    
    echo ""
    echo -e "${BLUE}🔧 Management:${NC}"
    echo "   • SSH:         ssh -i $SSH_KEY $SSH_USER@$SERVER_HOST"
    echo "   • Logs:        ssh -i $SSH_KEY $SSH_USER@$SERVER_HOST 'cd $REMOTE_PROJECT_DIR && tail -f *.log'"
    echo "   • Restart:     ./universal-deploy.sh restart"
    echo "   • Status:      ./universal-deploy.sh status"
}

# Main menu
show_menu() {
    echo -e "${BLUE}"
    echo "======================================================"
    echo "🚀 Universal Deploy - $PROJECT_NAME"
    echo "======================================================"
    echo -e "${NC}"
    echo ""
    echo "Available commands:"
    echo "  1) 🔧 Configure deployment"
    echo "  2) 🧪 Test connection"
    echo "  3) 📁 Sync code only"
    echo "  4) 🚀 Full deployment"
    echo "  5) ✅ Health check"
    echo "  6) 🔒 Setup SSL"
    echo "  7) 📊 Show status"
    echo "  8) 🔄 Restart services"
    echo "  9) ❌ Exit"
    echo ""
    read -p "Choose option [1-9]: " choice
    
    case $choice in
        1) create_deploy_config ;;
        2) load_config && test_connection ;;
        3) load_config && test_connection && sync_code ;;
        4) 
            load_config
            test_connection
            backup_deployment
            sync_code
            setup_remote
            deploy_app
            health_check
            setup_ssl
            show_final_info
            ;;
        5) load_config && test_connection && health_check ;;
        6) load_config && test_connection && setup_ssl ;;
        7) load_config && test_connection && ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" "cd $REMOTE_PROJECT_DIR && echo 'Project: $PROJECT_NAME' && ls -la" ;;
        8) load_config && test_connection && ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" "cd $REMOTE_PROJECT_DIR && $START_COMMAND" ;;
        9) log_info "Goodbye!" && exit 0 ;;
        *) log_error "Invalid option" && show_menu ;;
    esac
}

# Command line interface
case "$1" in
    "config") create_deploy_config ;;
    "test") load_config && test_connection ;;
    "sync") load_config && test_connection && sync_code ;;
    "deploy") 
        load_config
        test_connection
        backup_deployment
        sync_code
        setup_remote
        deploy_app
        health_check
        setup_ssl
        show_final_info
        ;;
    "health") load_config && test_connection && health_check ;;
    "ssl") load_config && test_connection && setup_ssl ;;
    "status") load_config && test_connection && ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" "cd $REMOTE_PROJECT_DIR && echo 'Status for: $PROJECT_NAME' && ls -la" ;;
    "restart") load_config && test_connection && ssh -i "$SSH_KEY" "$SSH_USER@$SERVER_HOST" "cd $REMOTE_PROJECT_DIR && $START_COMMAND" ;;
    *) show_menu ;;
esac
