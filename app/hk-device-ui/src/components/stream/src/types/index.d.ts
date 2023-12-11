/**
 * https://drylint.com/TypeScript/TS%E5%A3%B0%E6%98%8E%E6%96%87%E4%BB%B6d.ts.html#%E5%A3%B0%E6%98%8E%E5%91%BD%E5%90%8D%E7%A9%BA%E9%97%B4-%E5%85%A8%E5%B1%80%E5%AF%B9%E8%B1%A1-declare-namespace
 * 只要在声明文件中，出现了 import 或 export,
 * 那么这个声明文件就是模块声明文件，不再是全局声明文件。
 * 
 * 所有声明都属于局部声明，都只能在文件内部使用，或者通过 export 供外部使用。
 * 
 * 如果必须要在模块声明文件中，声明一些全局变量或全局类型，可以在 `declare global{}` 块中完成：
 */
interface PlayerOption{
  type:"MediaSource"|"Webcodec"|"SIMD"
  renderDom:"video"|"canvas" 
  ,videoBuffer:number// 缓存时长 s
  ,videoBufferDelay:number// 缓存延迟 s
  ,useCanvasRender:boolean
  ,useWebGPU?:boolean
  currentSource:number 

} 
interface stream_source{
  url:String
  name:String //// ['普清', '高清', '超清', '4K', '8K']
}
 
interface player_option{
  videoBuffer:number,
  videoBufferDelay:number,
  useCanvasRender:boolean,
  useWebGPU:boolean 
}

declare module  'JessibucaPro'
declare  class JessibucaPro { 
  constructor(opt:any)
  on(eventName:string, x:function)
} 