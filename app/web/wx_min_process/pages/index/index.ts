import { loginApi, LoginParams, UserInfo } from '@/services/api/user';
// 定义当前页面 Data 类型
interface IndexPageData {
    userInfo: UserInfo
    title: string
}
// 定义页面方法类型
interface IndexPageMethod {
    onTapItem(e: WechatMiniprogram.TouchEvent): void
    getUserInfo(): void
}
Page<IndexPageData, IndexPageMethod>({
    data: <IndexPageData>{  // <IndexPageData> 类型断言 等价于  {} as IndexPageData
        userInfo: {} as UserInfo, // TS 类型指定 
        title: '首页'
    },

    onLoad() {
        this.getUserInfo();
    },
    onTapItem(e: WechatMiniprogram.TouchEvent) {
        const id = e.currentTarget.dataset.id
        console.log(id)
    },
    // 获取用户信息
    async getUserInfo() {
        try {
            const res = await loginApi({ phone: '13800138000', code: '1234' });
            this.setData({
                userInfo: res.data,
            });
            console.log('用户信息：', this.data.userInfo);
        } catch (err) {
            console.error('请求失败：', err);
        }
    },
});