import {UserInfo} from './services/api/user';
// 全局 TS 类型
interface AppOption {
  globalData: {
    userInfo: null | UserInfo;
  };
}

App<AppOption>({
  globalData: {
    userInfo: null,
  },

  onLaunch() {
    console.log('小程序启动');
  },
});