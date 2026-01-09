/// <reference types="vite/client" />
//不生效
declare global {
  interface Window {
    _env_: {
      API_URL?: string
      [key: string]: any
    }
  }
}

