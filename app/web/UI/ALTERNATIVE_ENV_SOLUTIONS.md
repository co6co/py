# import.meta.env.VITE_SYSTEM_NAME 的替代方案

基于当前项目的技术栈（Vite + Vue 3 + TypeScript），以下是几种替代 `import.meta.env.VITE_SYSTEM_NAME` 的方案：

## 方案1：使用 Vite 的 define 配置

### 实现方式
在 `vite.config.ts` 中使用 `define` 选项注入全局变量：

```typescript
// vite.config.ts
import { loadEnv, defineConfig } from 'vite'

export default (option: { mode: string }) => {
  const env = loadEnv(option.mode, process.cwd())
  
  return defineConfig({
    // ... 其他配置
    define: {
      '__SYSTEM_NAME__': JSON.stringify(env.VITE_SYSTEM_NAME),
      '__SYSTEM_VERSION__': JSON.stringify(env.VITE_SYSTEM_VERSION),
      // 其他环境变量
    }
  })
}
```

### 使用方式
```typescript
// 在组件或钩子中直接使用
export default function () {
  const systeminfo = ref<ISystemInfo>({
    name: __SYSTEM_NAME__ || '管理系统',
    version: __SYSTEM_VERSION__ || '0.0.1',
    // ... 其他属性
  })
}
```

### 优缺点
- ✅ 构建时替换，运行时无需解析
- ✅ 直接访问，无需 `import.meta.env`
- ❌ 需要在 Vite 配置中手动维护

## 方案2：使用配置文件

### 实现方式
1. 创建环境特定的配置文件
2. 通过构建工具选择加载哪个配置文件

```typescript
// src/config/index.ts
// 默认配置
const defaultConfig = {
  name: '管理系统',
  version: '0.0.1',
  verifyType: 0,
  loginBgUrl: ''
}

// 根据环境导入不同配置
let envConfig = {}
if (import.meta.env.MODE === 'dev') {
  envConfig = await import('./dev').then(m => m.default)
} else if (import.meta.env.MODE === 'pro') {
  envConfig = await import('./pro').then(m => m.default)
} else if (import.meta.env.MODE === 'lbo') {
  envConfig = await import('./lbo').then(m => m.default)
}

export default {
  ...defaultConfig,
  ...envConfig
}
```

```typescript
// src/config/dev.ts
export default {
  name: '开发环境管理系统',
  version: '0.4.20250509',
  verifyType: 1,
  loginBgUrl: ''
}
```

### 使用方式
```typescript
// 在 useSystem.ts 中使用
import config from '@/config'

export default function () {
  const systeminfo = ref<ISystemInfo>({
    name: config.name,
    version: config.version,
    verifyType: config.verifyType,
    loginBgUrl: config.loginBgUrl
  })
  
  return { systeminfo }
}
```

### 优缺点
- ✅ 类型安全，IDE 友好
- ✅ 集中管理配置
- ✅ 支持复杂配置结构
- ❌ 需要为每个环境创建配置文件

## 方案3：使用全局变量注入

### 实现方式
在 `index.html` 中通过脚本注入全局变量：

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>管理系统</title>
    <script>
      // 通过构建工具动态替换这些值
      window.__APP_CONFIG__ = {
        SYSTEM_NAME: '%VITE_SYSTEM_NAME%',
        SYSTEM_VERSION: '%VITE_SYSTEM_VERSION%',
        SYSTEM_LOGIN_VERIFY_TYPE: '%VITE_SYSTEM_LOGIN_VERIFY_TYPE%',
        SYSTEM_LOGIN_BG_URL: '%VITE_SYSTEM_LOGIN_BG_URL%'
      }
    </script>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

然后在 Vite 配置中使用 `define` 替换这些值：

```typescript
// vite.config.ts
export default (option: { mode: string }) => {
  const env = loadEnv(option.mode, process.cwd())
  
  return defineConfig({
    // ... 其他配置
    define: {
      '%VITE_SYSTEM_NAME%': JSON.stringify(env.VITE_SYSTEM_NAME),
      '%VITE_SYSTEM_VERSION%': JSON.stringify(env.VITE_SYSTEM_VERSION),
      '%VITE_SYSTEM_LOGIN_VERIFY_TYPE%': JSON.stringify(env.VITE_SYSTEM_LOGIN_VERIFY_TYPE),
      '%VITE_SYSTEM_LOGIN_BG_URL%': JSON.stringify(env.VITE_SYSTEM_LOGIN_BG_URL)
    }
  })
}
```

### 使用方式
```typescript
// 在 useSystem.ts 中使用
declare global {
  interface Window {
    __APP_CONFIG__: {
      SYSTEM_NAME: string
      SYSTEM_VERSION: string
      SYSTEM_LOGIN_VERIFY_TYPE: string
      SYSTEM_LOGIN_BG_URL: string
    }
  }
}

export default function () {
  const systeminfo = ref<ISystemInfo>({
    name: window.__APP_CONFIG__?.SYSTEM_NAME || '管理系统',
    version: window.__APP_CONFIG__?.SYSTEM_VERSION || '0.0.1',
    verifyType: Number(window.__APP_CONFIG__?.SYSTEM_LOGIN_VERIFY_TYPE) || 0,
    loginBgUrl: window.__APP_CONFIG__?.SYSTEM_LOGIN_BG_URL || ''
  })
  
  return { systeminfo }
}
```

### 优缺点
- ✅ 全局可访问
- ✅ 支持动态注入
- ❌ 需要类型声明
- ❌ 依赖全局变量，不够模块化

## 方案4：使用环境特定的配置文件 + 构建时选择

### 实现方式
1. 创建不同环境的配置文件
2. 在构建时通过 Vite 的 `alias` 或其他方式选择加载哪个配置文件

```typescript
// src/config/envs/dev.ts
export default {
  name: '开发环境管理系统',
  version: '0.4.20250509',
  verifyType: 1,
  loginBgUrl: ''
}

// src/config/envs/pro.ts
export default {
  name: '生产环境管理系统',
  version: '0.4.20250509',
  verifyType: 1,
  loginBgUrl: ''
}

// src/config/index.ts
import envConfig from './envs/current'

const defaultConfig = {
  name: '管理系统',
  version: '0.0.1',
  verifyType: 0,
  loginBgUrl: ''
}

export default {
  ...defaultConfig,
  ...envConfig
}
```

然后在 Vite 配置中使用 `resolve.alias` 动态指向当前环境的配置文件：

```typescript
// vite.config.ts
export default (option: { mode: string }) => {
  return defineConfig({
    // ... 其他配置
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src'),
        './envs/current': resolve(__dirname, `src/config/envs/${option.mode}.ts`),
        vue: 'vue/dist/vue.esm-bundler.js'
      }
    }
  })
}
```

### 使用方式
```typescript
// 在 useSystem.ts 中使用
import config from '@/config'

export default function () {
  const systeminfo = ref<ISystemInfo>({
    name: config.name,
    version: config.version,
    verifyType: config.verifyType,
    loginBgUrl: config.loginBgUrl
  })
  
  return { systeminfo }
}
```

### 优缺点
- ✅ 类型安全，IDE 友好
- ✅ 集中管理配置
- ✅ 无需运行时解析
- ✅ 支持复杂配置结构
- ❌ 需要为每个环境创建配置文件

## 方案5：使用 dotenv 直接加载

### 实现方式
在代码中直接使用 dotenv 加载环境变量（需要安装 dotenv 依赖）：

```bash
npm install dotenv
```

```typescript
// src/utils/loadEnv.ts
import dotenv from 'dotenv'
import { resolve } from 'path'

export function loadEnv() {
  const envPath = resolve(process.cwd(), `.env.${process.env.NODE_ENV || 'dev'}`)
  const env = dotenv.config({ path: envPath })
  return env.parsed || {}
}
```

### 使用方式
```typescript
// 在 useSystem.ts 中使用
import { loadEnv } from '@/utils/loadEnv'

export default function () {
  const env = loadEnv()
  const systeminfo = ref<ISystemInfo>({
    name: env.VITE_SYSTEM_NAME || '管理系统',
    version: env.VITE_SYSTEM_VERSION || '0.0.1',
    verifyType: Number(env.VITE_SYSTEM_LOGIN_VERIFY_TYPE) || 0,
    loginBgUrl: env.VITE_SYSTEM_LOGIN_BG_URL || ''
  })
  
  return { systeminfo }
}
```

### 优缺点
- ✅ 无需依赖 Vite 的环境变量系统
- ✅ 可以在任何地方使用
- ❌ 需要额外安装依赖
- ❌ 运行时加载，有性能开销
- ❌ 仅适用于 Node.js 环境，浏览器环境需要其他处理

## 推荐方案

基于当前项目的技术栈和需求，推荐使用以下两种方案：

### 推荐方案1：方案2（配置文件）
- 类型安全，IDE 友好
- 集中管理配置
- 支持复杂配置结构
- 构建时静态分析，性能好

### 推荐方案2：方案1（Vite 的 define 配置）
- 构建时替换，运行时无需解析
- 直接访问，使用方便
- 无需额外依赖

## 实现示例：方案2（配置文件）

### 步骤1：创建配置文件结构
```
src/
└── config/
    ├── index.ts          # 主配置文件
    └── envs/
        ├── base.ts       # 基础配置
        ├── dev.ts        # 开发环境配置
        ├── pro.ts        # 生产环境配置
        └── lbo.ts        # LBO环境配置
```

### 步骤2：实现配置文件

```typescript
// src/config/envs/base.ts
export default {
  name: '管理系统',
  version: '0.0.1',
  verifyType: 0,
  loginBgUrl: ''
}
```

```typescript
// src/config/envs/dev.ts
import base from './base'

export default {
  ...base,
  name: '开发环境 - 服务管理系统',
  version: '0.4.20250509',
  verifyType: 1
}
```

```typescript
// src/config/envs/pro.ts
import base from './base'

export default {
  ...base,
  name: '生产环境 - 服务管理系统',
  version: '0.4.20250509',
  verifyType: 1
}
```

```typescript
// src/config/envs/lbo.ts
import base from './base'

export default {
  ...base,
  name: 'LBO环境 - 服务管理系统',
  version: '0.4.20250509',
  verifyType: 1
}
```

```typescript
// src/config/index.ts
import { loadEnv } from 'vite'

// 获取当前环境
const env = import.meta.env.MODE || 'dev'

// 动态导入对应环境的配置
let envConfig = {}
if (env === 'dev') {
  envConfig = await import('./envs/dev').then(m => m.default)
} else if (env === 'pro') {
  envConfig = await import('./envs/pro').then(m => m.default)
} else if (env === 'lbo') {
  envConfig = await import('./envs/lbo').then(m => m.default)
} else {
  envConfig = await import('./envs/base').then(m => m.default)
}

export default envConfig
```

### 步骤3：修改 useSystem.ts

```typescript
import { ref } from 'vue'
import config from '@/config'

export interface ISystemInfo {
  name: string
  version: string
  verifyType: number
  loginBgUrl: string
}

export default function () {
  const systeminfo = ref<ISystemInfo>({
    name: config.name,
    version: config.version,
    verifyType: config.verifyType,
    loginBgUrl: config.loginBgUrl
  })

  return { systeminfo }
}
```

通过以上方案，你可以替换掉当前使用的 `import.meta.env.VITE_SYSTEM_NAME`，实现更灵活、类型安全的配置管理。