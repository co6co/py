import ContextMenu from './contextMenu';

export type ContextMenuInstance = InstanceType<typeof ContextMenu>;
export { ContextMenu };

import type detail from './detail';
export type DetailInstance = InstanceType<typeof detail>;
export { default as Detail, type IDetails as Details } from './detail';

export {
	default as Dialog,
	type IDialogDataType as DialogDataType,
} from './dialog';
import type Dialog from './dialog';
export type DialogInstance = InstanceType<typeof Dialog>;

export * as DiaglogForm from './diaglogForm';
export * as DialogDetail from './dialogDetail';
export * as EnumSelect from './enumSelect';
export { default as Form, type IFormDataType } from './form';
import { default as Form2 } from './form';
export type FormInstance = InstanceType<typeof Form2>;

export * as Hello from './hello';
export * as IconSelect from './iconSelect';
export * as IntervalTime from './intervalTime';
export * from './logining';
