import { makeInstaller } from './make-installer';

import {
	ModifyMenu,
	ModifyRole,
	ModifyUser,
	ModifyUserGroup,
	ResetPwd,
	ResetPwdcopy,
} from './components';

import {
	MenuTreeView,
	RoleView,
	UserGroupTreeView,
	UserTableView,
} from './view';

import type { Plugin } from 'vue';
const components = [
	ModifyMenu,
	ModifyRole,
	ModifyUser,
	ModifyUserGroup,
	ResetPwd,
	ResetPwdcopy,

	MenuTreeView,
	RoleView,
	UserGroupTreeView,
	UserTableView,
] as Plugin[];
export default makeInstaller(components);
