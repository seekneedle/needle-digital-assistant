## Introduction
本项目提供数字人服务。

## Build docker image and upload to harbor
```sh
sh upload.sh
```

## Local run
添加环境变量
```
linux：export needle_assistant_uat="your_pwd" 和 needle_assistant_prod="your_prod_pwd"
IDE配置：Configuration->Enviroment->Environment variables中添加needle_assistant_uat=your_pwd 或者 needle_assistant_prod=your_prod_pwd
```

The app will be running on http://0.0.0.0:8501
```sh
python main.py
```
Run in background:
```sh
sh run.sh
```

Shutdown background:
```sh
sh shutdown.sh
```