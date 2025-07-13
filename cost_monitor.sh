#!/bin/bash

# macOS-use Cost Monitoring and Prevention Script

API_URL="http://localhost:8080"

echo "🏥 macOS-use Cost Monitor"
echo "========================="

# Check for stuck tasks
echo "🔍 Checking for stuck or long-running tasks..."
STUCK_TASKS=$(curl -s "$API_URL/tasks" | python3 -c "
import sys, json
data = json.load(sys.stdin)
running_tasks = [t for t in data.get('tasks', []) if t['status'] == 'running']
print(len(running_tasks))
")

if [ "$STUCK_TASKS" -gt 0 ]; then
    echo "⚠️  Found $STUCK_TASKS running task(s)"
    
    # Show task details
    curl -s "$API_URL/tasks" | python3 -c "
import sys, json
data = json.load(sys.stdin)
running_tasks = [t for t in data.get('tasks', []) if t['status'] == 'running']
for task in running_tasks:
    print(f\"  - Task {task['task_id']}: {task['status']}\")
"
    
    echo ""
    read -p "🛑 Clear stuck tasks? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🧹 Clearing cache and stuck tasks..."
        curl -X POST "$API_URL/cache/clear" > /dev/null 2>&1
        echo "✅ Cache cleared"
    fi
else
    echo "✅ No stuck tasks found"
fi

echo ""

# Check API health and settings
echo "📊 Current API settings:"
curl -s "$API_URL/config" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    settings = data.get('default_settings', {})
    print(f\"  Max steps: {settings.get('max_steps', 'unknown')}\")
    print(f\"  Max actions: {settings.get('max_actions', 'unknown')}\")
    print(f\"  Optimization: {data.get('optimization', 'unknown')}\")
except:
    print('  Could not fetch settings')
"

echo ""

# Suggest cost-saving measures
echo "💰 Cost-saving recommendations:"
echo "  1. Use gpt-3.5-turbo for testing (10x cheaper than gpt-4)"
echo "  2. Keep max_steps ≤ 15 for cost control"
echo "  3. Clear cache regularly: ./manage_services.sh clear-cache"
echo "  4. Monitor OpenAI usage: https://platform.openai.com/account/usage"
echo "  5. Set budget limits: https://platform.openai.com/account/billing/limits"

echo ""

# Test with cost-effective settings
echo "🧪 Want to run a cost-effective test?"
read -p "Run quick test with gpt-3.5-turbo? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Running cost-effective test..."
    RESULT=$(curl -s -X POST "$API_URL/quick-task?task=say%20cost%20test&provider=openai&model=gpt-3.5-turbo")
    TASK_ID=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('task_id', ''))")
    
    if [ ! -z "$TASK_ID" ]; then
        echo "✅ Test started (ID: $TASK_ID)"
        echo "💡 Check status: curl -s $API_URL/tasks/$TASK_ID"
    else
        echo "❌ Test failed to start"
    fi
fi

echo ""
echo "📝 Daily maintenance commands:"
echo "  ./manage_services.sh health        # Check system status"
echo "  ./manage_services.sh clear-cache   # Clear completed tasks"
echo "  ./cost_monitor.sh                  # Run this script"

echo ""
echo "✅ Cost monitoring complete!"
