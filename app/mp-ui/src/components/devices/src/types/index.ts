export  enum  DeviceState{
  loading,    //0
  Connected, //AI 和球机都连接 
  Disconected,
  AIConnected, //AI 连接 
  CameraConnected,//球机连接
} 
export interface Stream {
  url: string;
  valid:boolean;// 是否有效
  name: string;
}
export interface DeviceData {
  [key:string]:number|string|Stream[]|undefined;
  cameraType?: number;
  siteId: number; 
  sip: string;
  uuid: string;
  channel1_sip: string;
  channel2_sip: string;
  channel3_sip: string;
  channel4_sip: string;
  channel5_sip: string;
  channel6_sip: string;
  channel7_sip: string;
  channel8_sip: string;
  channel9_sip: string;
  channel10_sip: string; 
  createTime: string;
  createUser: number;
  id: number;
  innerConfigUrl: string;
  configUrl: string;
  innerIp: string;
  ip: string;
  name: string;
  no: number;
  poster: string; 
  streams: Stream[];
  
  //talkbackNo: number;
  updateTime: string;
  updateUser?: number;

  state?: DeviceState;
  statueComponent?:any;
}
export interface BoxDevice {
  mac:string;
  createUser:string;
  license:string;
  updateUser:string;
  innerIp:string;
  talkbackNo:number;
  createTime:string;
  siteId:number; 
  updateTime:string;
  id:number;
  cpuNo:string; 
  uuid:string;
  /*
  sip:string;
  channel2_sip:string;
  channel1_sip:string;
  channel3_sip:string;
  */
  ip:string; 
  name:string;
  innerConfigUrl:string;
  configUrl:string;
  sshConfigUrl?:string;
}

export interface CurrentItem{
  box?:BoxDevice
  device?:DeviceData
}

export interface Site{
  id:number;
  name:string;
  deviceCode:string;
  postionInfo:string;
  deviceDesc:string;
  createTime:string;
  updateTime:string;
  
  box?:BoxDevice;
  devices?:Array< DeviceData>; 
  state?: DeviceState;
  statueComponent?:any;
  device?:DeviceData;
  
} 

export interface talkState{
  state:number;
  stateDesc:string;
  talkNo:number 
}
 
 //xtalk 
 //用到再加
export interface talkerMessageData{
	SessionId:string;
}