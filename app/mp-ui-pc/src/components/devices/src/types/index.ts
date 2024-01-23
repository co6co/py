export  enum  DeviceState{
  loading,
  Connected,
  Disconected
} 
export interface Stream {
  url: string;
  name: string;
}
export interface DeviceData {
  cameraType?: number;
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
  configUrl: string;
  createTime: string;
  createUser: number;
  id: number;
  innerConfigUrl: string;
  innerIp: string;
  ip: string;
  name: String;
  no: number;
  poster: string;
  sip: string;
  siteId: number; 
  streams: Stream;
  talkbackNo: number;
  updateTime: string;
  updateUser?: number;
  uuid: string;
  state?: DeviceState;
  statueComponent?:any;
}
export interface BoxDevice {
  mac:string;
  createUser:string;
  license:string;
  updateUser:string;
  innerIp:string;
  talkbackNo:string;
  createTime:string;
  siteId:number;
  channel1_sip:string;
  updateTime:string;
  id:number;
  cpuNo:string;
  channel2_sip:string;
  uuid:string;
  sip:string;
  channel3_sip:string;
  ip:string;
  innerConfigUrl:string;
  name:string;
  configUrl:string;
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
 

export interface talkState{
  state:number;
  stateDesc:string;
  talkNo:number 
}
 