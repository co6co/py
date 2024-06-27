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
} from './components';
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
] as Plugin[];
export default makeInstaller(components);
