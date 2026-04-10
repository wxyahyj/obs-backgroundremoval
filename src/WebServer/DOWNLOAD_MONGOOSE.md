# Mongoose 库下载说明

## 下载方式

### 方式1：直接下载（推荐）
从以下地址下载两个文件：
- https://raw.githubusercontent.com/cesanta/mongoose/master/mongoose.h
- https://raw.githubusercontent.com/cesanta/mongoose/master/mongoose.c

下载后放入此目录。

### 方式2：使用Git
```bash
cd e:\obs-heji\obs-backgroundremoval\src\WebServer
git clone --depth 1 https://github.com/cesanta/mongoose.git temp
cp temp/mongoose.h .
cp temp/mongoose.c .
rm -rf temp
```

### 方式3：使用curl（Windows PowerShell）
```powershell
cd e:\obs-heji\obs-backgroundremoval\src\WebServer
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/cesanta/mongoose/master/mongoose.h" -OutFile "mongoose.h"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/cesanta/mongoose/master/mongoose.c" -OutFile "mongoose.c"
```

## 版本信息
- 推荐版本：最新版（master分支）
- 许可证：MIT License
- 官网：https://mongoose.ws/
- GitHub：https://github.com/cesanta/mongoose

## 文件说明
- `mongoose.h` - 头文件，包含API声明
- `mongoose.c` - 实现文件，包含完整实现

## 集成方式
这两个文件只需要添加到项目中即可，无需额外配置。
