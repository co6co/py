export enum Expire {
	expire = '__expire__',
	permanent = 'permanent',
}

export type Key = string; //key类型
export type expire = Expire.permanent | number; //有效期类型
export interface Data<T> {
	//格式化data类型
	value: T;
	[Expire.expire]: Expire.expire | number;
}
 
export interface StorageCls {
	set: <T>(key: Key, value: T, expire:expire ) => void;
	get: <T>(key: Key) =>  T | null;
	remove: (key: Key) => void;
	clear: () => void;
}

export class Storage implements StorageCls {
	//存储接受 key value 和过期时间 默认永久
	public set<T = any>(key: Key, value: T, expire:expire= Expire.permanent) { 
		//格式化数据
		const data = {
			value,
			[Expire.expire]: typeof( expire)=="number"? new Date().getTime()+expire*1000:expire,
		};
		//存进去
		localStorage.setItem(key, JSON.stringify(data));
	}

	public get<T = any>(key: Key): T | null {
		const value = localStorage.getItem(key);
		//读出来的数据是否有效
		if (value) {
			const obj: Data<T> = JSON.parse(value);
			const now = new Date().getTime();
			//有效并且是数组类型 并且过期了 进行删除和提示
			if (typeof obj[Expire.expire] == 'number' && obj[Expire.expire] < now) {
				this.remove(key);
				return null;
			} else {
				//否则成功返回
				return   obj.value
				 
			}
		} else {
			//否则key值无效
			//console.warn(`key:'${key}'值无效'`);
			return   null 
			 
		}
	}
	//删除某一项
	public remove(key: Key) {
		localStorage.removeItem(key);
	}
	//清空所有值
	public clear() {
		localStorage.clear();
	}
}
