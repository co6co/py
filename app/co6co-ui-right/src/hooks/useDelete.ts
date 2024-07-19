import { showLoading, closeLoading, type IResponse } from 'co6co';
import { ElMessageBox, ElMessage } from 'element-plus';
export default function (
	del_svc: (id: number) => Promise<IResponse>,
	bck?: () => void
) {
	const onDelete = (pk: any, name: string) => {
		// 二次确认删除
		ElMessageBox.confirm(`确定要删除"${name}"吗？`, '提示', {
			type: 'warning',
		})
			.then(() => {
				showLoading();
				del_svc(pk)
					.then((res) => {
						ElMessage.success(res.message || '删除成功');
						if (bck) bck();
					})
					.finally(() => {
						closeLoading();
					});
			})
			.catch(() => {});
	};
	return { onDelete };
}
