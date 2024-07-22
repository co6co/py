import { showLoading, closeLoading, type IResponse } from 'co6co';
import { ElMessageBox, ElMessage } from 'element-plus';
export default function (
	del_svc: (id: number) => Promise<IResponse>,
	bck?: () => void
) {
	const deleteSvc = (pk: any, name?: string) => {
		if (!name) name = '数据';
		// 二次确认删除
		ElMessageBox.confirm(`确定要删除"${name}"吗？`, '提示', {
			type: 'warning',
		})
			.then(() => {
				showLoading();
				return del_svc(pk);
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
