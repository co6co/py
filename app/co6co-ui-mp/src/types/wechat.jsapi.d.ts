declare global {
    interface WX extends WechatJSAPI {}
  
    interface WechatJSAPI {
      config(config: WechatJSAPIConfig): void;
      ready(callback: () => void): void;
      error(callback: (err: any) => void): void;
      onMenuShareAppMessage(options: WechatJSAPIOnMenuShareOptions): void;
      onMenuShareTimeline(options: WechatJSAPIOnMenuShareOptions): void;
    }
  
    interface WechatJSAPIConfig {
      debug?: boolean;
      appId: string;
      timestamp: number;
      nonceStr: string;
      signature: string;
      jsApiList: string[];
    }
  
    interface WechatJSAPIOnMenuShareOptions {
      title?: string;
      desc?: string;
      link?: string;
      imgUrl?: string;
      success?: () => void;
      cancel?: () => void;
      fail?: (res: any) => void;
    }
  }
  
  declare const wx: WechatJSAPI;
