/// <reference types="miniprogram-api-typings" />
import request from '@/services/request';
import type { AxiosRequestConfig } from 'axios';

// 统一中间层封装
const api = {
    /**
     * GET 请求
     */
    get<T>(
        url: string,
        params?: any,
        config?: AxiosRequestConfig
    ): Promise<T> {
        wx.showLoading({ title: '加载中...' }); 
        return request
            .get<T>(url, { params, ...config })
            .finally(() => {
                wx.hideLoading();
            });
    },

    /**
     * POST 请求
     */
    post<T>(
        url: string,
        data?: any,
        config?: AxiosRequestConfig
    ): Promise<T> {
        wx.showLoading({ title: '加载中...' });

        return request
            .post<T>(url, data, config)
            .finally(() => {
                wx.hideLoading();
            });
    },

    put<T>(url: string, data?: any): Promise<T> {
        wx.showLoading({ title: '加载中...' });
        return request.put<T>(url, data).finally(wx.hideLoading);
    },

    delete<T>(url: string): Promise<T> {
        wx.showLoading({ title: '加载中...' });
        return request.delete<T>(url).finally(wx.hideLoading);
    },
};

export default api;