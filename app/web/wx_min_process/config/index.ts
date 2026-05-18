// config/index.ts
const isDev = true; // 开发时 true，发布时改成 false

const config = {
  // 开发环境
  development: {
    baseUrl: 'https://api-dev.xxx.com',
  },
  // 生产环境
  production: {
    baseUrl: 'https://api.xxx.com',
  },
};

// 自动根据 isDev 切换环境
export default isDev ? config.development : config.production;