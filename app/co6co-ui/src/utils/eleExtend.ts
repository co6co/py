import { ElMessageBox } from 'element-plus';
declare type MessageType = '' | 'success' | 'warning' | 'info' | 'error';
declare type ConfirmType = { tipName: string; type: { type: MessageType } };

export const warningArgs: ConfirmType = {
	tipName: '提示',
	type: { type: 'warning' },
};

export const EleConfirm = (message, { tipName, type }: ConfirmType) => {
	return ElMessageBox.confirm(message, tipName, type);
};
