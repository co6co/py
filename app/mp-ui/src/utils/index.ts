import JSONbig  from 'json-bigint' 
import {  dayjs } from 'element-plus';
export const str2Obj=(str:string)=>{
    return JSONbig.parse( str)
}

/// 创建开始结束时间
export const createStateEndDatetime=(type:number, beforeHour:number)=>{ 
	let endDate=null
	let startDate=null
	switch (type){
		case 0:
			endDate=new Date();
			const times=endDate.getTime()-beforeHour*3600*1000
			startDate=new Date(times)
			break
		case 1:
			startDate=new Date( dayjs(new Date()) .format('YYYY/MM/DD'))
			endDate=startDate.getTime()+24*3600*1000-1000
			break
		default:
			startDate=new Date( dayjs(new Date()) .format('YYYY/MM/DD'))
			endDate=startDate.getTime()+24*3600*1000 -1000
			break
	}  
	return [dayjs(startDate) .format('YYYY-MM-DD HH:mm:ss'),dayjs(endDate) .format('YYYY-MM-DD HH:mm:ss')]
}


//生成随机字符串
export const randomString=(len:number, chars:string='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') =>{
    var result = '';
    for (var i = len; i > 0; --i) result += chars[Math.floor(Math.random() * chars.length)];
    return result;
} 
 