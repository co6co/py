/// <reference types="miniprogram-api-typings" />
import axios from "axios";
import config from "@/config";
import type { AxiosRequestConfig } from "axios";

const request = axios.create({
  baseURL: config.baseUrl,
  timeout: 15000,
});

// 请求拦截
request.interceptors.request.use((cfg) => {
  const token = wx.getStorageSync("token");
  if (token) {
    cfg.headers.Authorization = `Bearer ${token}`;
  }
  return cfg;
});

// 响应拦截：直接返回 data，剥离 AxiosResponse
request.interceptors.response.use(
  (res) => {
    return res.data; // ✅ 这里是关键！
  },
  (err) => {
    wx.showToast({ title: "网络异常", icon: "none" });
    return Promise.reject(err);
  }
);

// 👇 👇 👇 核心修复：手动封装返回 Promise<T>
export default {
  get<T>(url: string, config?: AxiosRequestConfig) {
    return request.get(url, config) as Promise<T>;
  },
  post<T>(url: string, data?: any, config?: AxiosRequestConfig) {
    return request.post(url, data, config) as Promise<T>;
  },
  put<T>(url: string, data?: any, config?: AxiosRequestConfig) {
    return request.put(url, data, config) as Promise<T>;
  },
  delete<T>(url: string, config?: AxiosRequestConfig) {
    return request.delete(url, config) as Promise<T>;
  },
};