import { showLoading, closeLoading, type IResponse } from 'co6co';
import { ElMessageBox, ElMessage, MessageBoxData } from 'element-plus';
export default function (
	del_svc: (id: number, reson?: string) => Promise<IResponse>,
	bck?: () => void
) {
	const deleteSvc = (
		pk: any,
		name?: string,
		hasReason?: boolean,
		messageBoxPromise?: Promise<MessageBoxData>
	) => {
		if (!name) name = '数据';
		// 二次确认 没有原因 删除
		if (!messageBoxPromise && !hasReason) {
			messageBoxPromise = ElMessageBox.confirm(
				`确定要删除"${name}"吗？`,
				'提示',
				{
					type: 'warning',
				}
			);
		} else if (!messageBoxPromise && hasReason) {
			//有原因删除
			messageBoxPromise = ElMessageBox.prompt(`请输入删除${name}原因`, 'Tip', {
				confirmButtonText: '确认',
				cancelButtonText: '取消',
				/*inputPattern: /d?/,
				inputErrorMessage: '无效的输入',
				*/
			});
		}

		messageBoxPromise
			?.then(({ value }) => {
				showLoading();
				return del_svc(pk, value);
			})
			.then((res) => {
				ElMessage.success(res.message || '删除成功');
				if (bck) bck();
			})
			.finally(() => {
				closeLoading();
			});
	};
	return { deleteSvc };
}
