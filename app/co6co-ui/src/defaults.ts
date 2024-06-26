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
] as Plugin[];
export default makeInstaller(components);
