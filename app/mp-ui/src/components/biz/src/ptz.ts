import {ref} from 'vue'
import { number2hex } from '../../../utils';
import * as p from '../../../components/stream/src/types/ptz';

const str2ByteArray = (ptzCmd: string) => {
	if (ptzCmd && ptzCmd.length != 16) console.error('命令不符规格');
	let arr: number[] = [0, 0, 0, 0, 0, 0, 0, 0];
	for (let i = 0, j = 0; i < ptzCmd.length; i += 2, j++) {
		//console.info(`${ptzCmd[i]}${ptzCmd[i+1]}`)
		arr[j] = parseInt(`${ptzCmd[i]}${ptzCmd[i + 1]}`, 16);
	}
	return arr;
};
const checkSum = (ptzCmd: number[]) => {
	//检测检验和
	let check: number = 0;
	for (let i = 0; i <= 6; i++) {
		check += ptzCmd[i];
	}
	if (ptzCmd[7] == check % 256) return true;
	return false;
};
//检验和
const genPtzCommd = (ptzCmd: number[]) => {

	let check: number = 0;
	let cmdstr: string = '';
	for (let i = 0; i <= 6; i++) {
		check += ptzCmd[i];
		cmdstr += number2hex(ptzCmd[i], 2);
	}
	ptzCmd[7] = check % 256;
	cmdstr += number2hex(ptzCmd[7], 2);
	return cmdstr;
};
export const createPtzCmd = (speed: number, type: p.ptz_type, name: p.ptz_name) => {
	let ptzCmd: number[] = [0xa5, 0x0f, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
	if (type == 'starting') {
		switch (name) {
			case 'up':
				ptzCmd[3] = 0x08;
				ptzCmd[5] = speed & 0xff;
				break;
			case 'down':
				ptzCmd[3] = 0x04;
				ptzCmd[5] = speed & 0xff;
				break;
			case 'right':
				ptzCmd[3] = 0x01;
				ptzCmd[4] = speed & 0xff;
				break;
			case 'left':
				ptzCmd[3] = 0x02;
				ptzCmd[4] = speed & 0xff;
				break;
			case 'zoomout':
				ptzCmd[3] = 0x20;
				//低四位转高四位,后取高四位
				ptzCmd[6] =speed<<4 & 0xf0; 
				if (speed>15)ptzCmd[6]=0xf0
				break;
			case 'zoomin':
				ptzCmd[3] = 0x10; 
				//低四位转高四位,后取高四位
				ptzCmd[6] =speed<<4 & 0xf0; 
				if (speed>15)ptzCmd[6]=0xf0
				break;
		}
	} else {
		//0
	}
	return genPtzCommd(ptzCmd);
};
export const snRef=ref(0)
export const generatePtzXml = (
	sip: string,
	speed: number,
	type: p.ptz_type,
	name: p.ptz_name
) :{sn:number,xml:string}=> {
	const cmdstr = createPtzCmd(speed, type, name);
	let time=new Date()
	snRef.value+=1
	//let sn = time.getSeconds()*1000+ time.getMilliseconds();
	let sn=snRef.value;
	/*
	let xml = `
    <?xml version="1.0" encoding="UTF-8"?>
    <Control>
        <CmdType>DeviceControl</CmdType>
        <SN>${sn}</SN>
        <DeviceID>${sip}</DeviceID>
        <PTZCmd>${cmdstr}</PTZCmd>
    </Control> 
    `;
	xml=`
	<?xml version="1.0" encoding="GB2312"?>
	<Control>
		<CmdType>DeviceControl</CmdType>
		<SN>${sn}</SN>
		<DeviceID>${sip}</DeviceID>
		<PTZCmd>${cmdstr}</PTZCmd>
		<Info>
			<ControlPriority>5</ControlPriority>
		</Info>
	</Control>
	`
	*/
	let xml=`<?xml version="1.0"?>
    <Control>
        <CmdType>DeviceControl</CmdType>
        <SN>${sn}</SN>
        <DeviceID>${sip}</DeviceID>
        <PTZCmd>${cmdstr}</PTZCmd>
		<Info>
			<ControlPriority>5</ControlPriority>
		</Info>
    </Control>  
	`
	return {sn:sn,xml:xml};
};

export const testPtzCmdStr = (ptzCmd: string) => {
    console.warn(ptzCmd)
	let arr: number[] = str2ByteArray(ptzCmd);
	let result = checkSum(arr);
	if (!result) console.warn(`检测位[${arr[7]}] 不正确`);
	let d4 = arr[3];
	let zoomOut = 32; //缩小
	let zoomInt = 16; //放大
	let up = 8;
	let down = 4;
	let left = 2;
	let right = 1;

	let iszoomOut = d4 & zoomOut;
	let iszoomInt = d4 & zoomInt;
	let zoomSeep = (arr[6] & 0xf0) >> 4;
	let isup = d4 & up;
	let isDown = d4 & down;
	let isleft = d4 & left;
	let isright = d4 & right;

	let d5 = arr[4]; //水平速度
	let d6 = arr[5]; //垂直速度
	var value = d4.toString(2);
	console.info('4:', d4, '=>', value);
	console.info(
		'zoom 缩小：',
		iszoomOut,
		'zoom 放大：',
		iszoomInt,
		'速度',
		zoomSeep
	);
	console.info('上：', isup, '下：', isDown);
	console.info('左：', isleft, '右：', isright);
	console.info('水平速度：', d5, '垂直速度：', d6);

	//Iris
	let fiIriBig = 72; //FI 光圈小
	let fiIriSmail = 68; //FI 光圈大
	//Focus
	let FocusIn = 66; //近
	let FocusOut = 65; //远
	d5 = arr[4]; //聚焦速度
	d6 = arr[4]; //光圈速度
};
