#!/bin/bash

# 查找监听8510端口的进程PID
pid=$(lsof -i :8510 | awk 'NR>1 {print $2}')

# 如果找到了PID，则尝试杀死进程
if [ -n "$pid" ]; then
    echo "Stopping application with PID $pid..."
    kill $pid
    echo "Application stopped."
else
    echo "No process found listening on port 8510."
fi