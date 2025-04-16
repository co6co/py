import { makeInstaller } from './make-installer';
import { views } from '@/views';

//只在本模块中使用的组件makeInstaller

//不用安装 也可以用
//测试了 dictSelect
/*
import {
	ModifyMenu,
	ModifyRole,
	ModifyUser,
	ModifyUserGroup,
	ResetPwd,
} from './components';
 const components = [
	ModifyMenu,
	ModifyRole,
	ModifyUser,
	ModifyUserGroup,
	ResetPwd,
];
*/
import type { Plugin } from 'vue';

const _view: Array<any> = [];
Object.keys(views).forEach((key) => {
	_view.push(views[key]);
});

export default makeInstaller([..._view] as Plugin[]);
