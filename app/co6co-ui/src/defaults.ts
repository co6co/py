import { makeInstaller } from './make-installer';
import {
	Detail,
	Dialog,
	DialogForm,
	DialogDetail,
	EnumSelect,
	Form,
	Hello,
	IconSelect,
	IntervalTime,
	Associated,
	ViewSelect,
} from './components';
//import { MenuTreeView } from './view';
import type { Plugin } from 'vue';
const components = [
	Detail,
	Dialog,
	DialogForm,
	DialogDetail,
	EnumSelect,
	Form,
	Hello,
	IconSelect,
	IntervalTime,
	Associated,
	ViewSelect,
] as Plugin[];
export default makeInstaller(components);
