1. 创建项目

```
npm create vite
>projec_name
>vue
>ts

```

2. 根目录建立 ts 声明文件

```
//env.d.ts
declare module "*.vue" {
  import { DefineComponent } from "vue"
  const component: DefineComponent<{}, {}, any>
  export default component
}
declare interface Window {
  Vue: any,
}
```

## 2.1 在 tsconfig.json 中的 include 加入申明文件，不然在 ts 文件导入 vue 模块会报错

```
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "preserve",
    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["packages/**/*.ts", "packages/**/*.d.ts", "packages/**/*.tsx", "packages/**/*.vue", "./*.d.ts"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

# 3. 改造目录

- 将 src-重命名->examples 目录，用作示例
- 创建新的 src 存放源码
- 启动项目时，默认入口`src/main.ts`,在 index.html 中把`/src/main.ts`改为`/examples/main.ts`

# 4. 组件开发

在新 src 目录开发组件，创建组件每个目录为一个组件 el-button 目录中有 src 目录和 index.ts 文件

# 5. 组件全局注册

```
//src\index.ts
// 导入单个组件
import elButton from './el-button/index'
import elButtonPlus from './el-button-plus/index'
// 以数组的结构保存组件，便于遍历
const components = [
  elButton,
  elButtonPlus
]
// 用于按需导入
export {
  elButton,
  elButtonPlus
}
// 定义 install 方法
const install = function (Vue: any) {
  if ((install as any).installed) return;
  (install as any).installed = true
  // 遍历并注册全局组件
  components.map(component => {
      Vue.component(component.name, component)
  })
}
if (typeof window !== 'undefined' && window.Vue) {
  install(window.Vue)
}
export default {
  // 导出的对象必须具备一个 install 方法
  install,
}
```

# 6. 本地导入组件测试

将 src 为基础导入组件

```
在examples 中App.vue 导入测试
```

# 7. 编写 package.json 文件

```

  "files": [
    "dist/*",
    "co6co-demo.d.ts"
  ],
  "main": "dist/co6co-demo.umd.js",
  "module": "dist/co6co-demo.es.js",
```

# 8. 编写 vite.config.ts 文件

```
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import vueJsx from '@vitejs/plugin-vue-jsx';
import path from 'path';
const resolve = (dir: string) => path.join(__dirname, dir);

// https://vitejs.dev/config/
export default defineConfig({
	plugins: [vue(), vueJsx()],
	resolve: {
		alias: {
			'@': resolve('examples'),
		},
	},
	build: {
		rollupOptions: {
			// 请确保外部化那些你的库中不需要的依赖
			external: ['vue'],
			output: {
				// 在 UMD 构建模式下为这些外部化的依赖提供一个全局变量
				globals: {
					vue: 'Vue',
				},
			},
		},
		lib: {
			entry: 'src/index.ts',
			name: 'co6co-demo',
			fileName: (format) => `co6co-demo.${format}.js`,
		},
	},
});


```

#9 `pnpm run build `

# 10. 本地模拟

以发布的 dict 为为主导入组件测试模拟
