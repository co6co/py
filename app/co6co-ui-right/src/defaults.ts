import { makeInstaller } from './make-installer';
import { views } from '@/views';

//只在本模块中使用的组件makeInstaller
/*
import {
	ModifyMenu,
	ModifyRole,
	ModifyUser,
	ModifyUserGroup,
	ResetPwd,
} from './components';
 */

import type { Plugin } from 'vue';
const _view: Array<any> = [];
Object.keys(views).forEach((key) => {
	_view.push(views[key]);
});
/**
const components = [
	ModifyMenu,
	ModifyRole,
	ModifyUser,
	ModifyUserGroup,
	ResetPwd,
	ResetPwdcopy,
];
 */
export default makeInstaller(_view as Plugin[]);
