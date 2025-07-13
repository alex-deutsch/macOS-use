#!/bin/bash

# macOS-use Services Management Script

GRADIO_SERVICE="com.macos-use.gradio"
API_SERVICE="com.macos-use.api"
GRADIO_PLIST="$HOME/Library/LaunchAgents/${GRADIO_SERVICE}.plist"
API_PLIST="$HOME/Library/LaunchAgents/${API_SERVICE}.plist"

start_service() {
    service_name=$1
    plist_path=$2
    echo "Starting $service_name..."
    launchctl load "$plist_path"
}

stop_service() {
    service_name=$1
    plist_path=$2
    echo "Stopping $service_name..."
    launchctl unload "$plist_path" 2>/dev/null
}

check_service_status() {
    service_name=$1
    if launchctl list | grep -q "$service_name"; then
        PID=$(launchctl list | grep "$service_name" | awk '{print $1}')
        echo "✅ $service_name is running (PID: $PID)"
        return 0
    else
        echo "❌ $service_name is not running"
        return 1
    fi
}

case "$1" in
    start)
        if [ "$2" = "gradio" ]; then
            start_service "$GRADIO_SERVICE" "$GRADIO_PLIST"
            echo "Gradio Web UI available at: http://localhost:7860"
        elif [ "$2" = "api" ]; then
            start_service "$API_SERVICE" "$API_PLIST"
            echo "REST API available at: http://localhost:8080"
            echo "API Documentation: http://localhost:8080/docs"
        else
            start_service "$GRADIO_SERVICE" "$GRADIO_PLIST"
            start_service "$API_SERVICE" "$API_PLIST"
            echo "Both services started!"
            echo "🌐 Gradio Web UI: http://localhost:7860"
            echo "🔌 REST API: http://localhost:8080"
            echo "📚 API Docs: http://localhost:8080/docs"
        fi
        ;;
    stop)
        if [ "$2" = "gradio" ]; then
            stop_service "$GRADIO_SERVICE" "$GRADIO_PLIST"
        elif [ "$2" = "api" ]; then
            stop_service "$API_SERVICE" "$API_PLIST"
        else
            stop_service "$GRADIO_SERVICE" "$GRADIO_PLIST"
            stop_service "$API_SERVICE" "$API_PLIST"
            echo "Both services stopped."
        fi
        ;;
    restart)
        if [ "$2" = "gradio" ]; then
            stop_service "$GRADIO_SERVICE" "$GRADIO_PLIST"
            start_service "$GRADIO_SERVICE" "$GRADIO_PLIST"
            echo "Gradio Web UI available at: http://localhost:7860"
        elif [ "$2" = "api" ]; then
            stop_service "$API_SERVICE" "$API_PLIST"
            start_service "$API_SERVICE" "$API_PLIST"
            echo "REST API available at: http://localhost:8080"
        else
            stop_service "$GRADIO_SERVICE" "$GRADIO_PLIST"
            stop_service "$API_SERVICE" "$API_PLIST"
            start_service "$GRADIO_SERVICE" "$GRADIO_PLIST"
            start_service "$API_SERVICE" "$API_PLIST"
            echo "Both services restarted!"
            echo "🌐 Gradio Web UI: http://localhost:7860"
            echo "🔌 REST API: http://localhost:8080"
        fi
        ;;
    status)
        echo "=== macOS-use Services Status ==="
        check_service_status "$GRADIO_SERVICE"
        check_service_status "$API_SERVICE"
        echo ""
        echo "🌐 Gradio Web UI: http://localhost:7860"
        echo "🔌 REST API: http://localhost:8080"
        echo "📚 API Documentation: http://localhost:8080/docs"
        echo ""
        echo "📄 Logs:"
        echo "  Gradio: /tmp/macos-use-gradio.log"
        echo "  API: /tmp/macos-use-api.log"
        ;;
    logs)
        if [ "$2" = "gradio" ]; then
            echo "=== Gradio Logs ==="
            tail -20 /tmp/macos-use-gradio.log
        elif [ "$2" = "api" ]; then
            echo "=== API Logs ==="
            tail -20 /tmp/macos-use-api.log
        else
            echo "=== Gradio Logs ==="
            tail -10 /tmp/macos-use-gradio.log
            echo ""
            echo "=== API Logs ==="
            tail -10 /tmp/macos-use-api.log
        fi
        ;;
    errors)
        if [ "$2" = "gradio" ]; then
            echo "=== Gradio Errors ==="
            if [ -f /tmp/macos-use-gradio-error.log ]; then
                tail -20 /tmp/macos-use-gradio-error.log
            else
                echo "No error log found."
            fi
        elif [ "$2" = "api" ]; then
            echo "=== API Errors ==="
            if [ -f /tmp/macos-use-api-error.log ]; then
                tail -20 /tmp/macos-use-api-error.log
            else
                echo "No error log found."
            fi
        else
            echo "=== Gradio Errors ==="
            if [ -f /tmp/macos-use-gradio-error.log ]; then
                tail -10 /tmp/macos-use-gradio-error.log
            else
                echo "No gradio error log found."
            fi
            echo ""
            echo "=== API Errors ==="
            if [ -f /tmp/macos-use-api-error.log ]; then
                tail -10 /tmp/macos-use-api-error.log
            else
                echo "No API error log found."
            fi
        fi
        ;;
    open)
        if [ "$2" = "api" ]; then
            echo "Opening API documentation in browser..."
            open http://localhost:8080/docs
        else
            echo "Opening Gradio web UI in browser..."
            open http://localhost:7860
        fi
        ;;
    test)
        echo "Testing API with simple task..."
        curl -X POST "http://localhost:8080/quick-task?task=say%20hello" 2>/dev/null | python3 -m json.tool
        ;;
    clear-cache)
        echo "Clearing API cache..."
        curl -X POST "http://localhost:8080/cache/clear" 2>/dev/null | python3 -m json.tool
        ;;
    health)
        echo "Checking API health..."
        curl -s "http://localhost:8080/health" | python3 -m json.tool
        ;;
    reset-accessibility)
        echo "Resetting accessibility state..."
        curl -X POST "http://localhost:8080/accessibility/reset" 2>/dev/null | python3 -m json.tool
        ;;
    ollama-status)
        echo "Checking Ollama status..."
        curl -s "http://localhost:8080/ollama/status" | python3 -m json.tool
        ;;
    ollama-models)
        echo "Listing available Ollama models..."
        curl -s "http://localhost:8080/ollama/models" | python3 -m json.tool
        ;;
    test-ollama)
        if [ -z "$2" ]; then
            echo "Usage: $0 test-ollama [model_name]"
            echo "Example: $0 test-ollama llama2"
            exit 1
        fi
        echo "Testing Ollama with model $2..."
        curl -X POST "http://localhost:8080/quick-task?task=say%20hello%20from%20ollama&provider=ollama&model=$2" 2>/dev/null | python3 -m json.tool
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|errors|open|test|clear-cache|health|reset-accessibility|ollama-status|ollama-models|test-ollama} [gradio|api]"
        echo ""
        echo "Service Commands:"
        echo "  start [service]       - Start service(s) (both if no service specified)"
        echo "  stop [service]        - Stop service(s)"
        echo "  restart [service]     - Restart service(s)"
        echo "  status                - Check services status"
        echo "  logs [service]        - Show recent logs"
        echo "  errors [service]      - Show recent errors"
        echo "  open [api]            - Open web UI/API docs in browser"
        echo ""
        echo "API Commands:"
        echo "  test                  - Test API with simple request"
        echo "  clear-cache           - Clear API cache to fix stale elements"
        echo "  health                - Check API health and statistics"
        echo "  reset-accessibility   - Reset accessibility state (fixes stale elements)"
        echo ""
        echo "Ollama Commands:"
        echo "  ollama-status         - Check if Ollama is running"
        echo "  ollama-models         - List available Ollama models"
        echo "  test-ollama [model]   - Test Ollama with specific model"
        echo ""
        echo "Services:"
        echo "  gradio            - Web UI (port 7860)"
        echo "  api               - Enhanced REST API v2 with Ollama support (port 8080)"
        exit 1
        ;;
esac
