import { defineComponent, ref, PropType } from 'vue';
import { ElButton, ElInput, ElMessageBox } from 'element-plus';
import { UploadFilled } from '@element-plus/icons-vue';
import { IResponse } from 'co6co';
type DataApi = (data: FormData) => Promise<IResponse>;

export default defineComponent({
	name: 'UploadFile',
	props: {
		txt: {
			type: String,
			default: '上传文件',
		},
		//是否需要确认上传
		confirm: {
			type: [Boolean, Object] as PropType<boolean | Promise<any>>,
			default: true,
		},
		accept: {
			type: String,
			default: '', //.xlsx,.xls
		},

		uploadApi: {
			type: Function as PropType<DataApi>,
			required: true,
		},
	},
	emits: {
		uploadBefore: (data: FormData, file: File) => true,
		success: (data: IResponse<any>) => true,
		error: (data: IResponse<any> | Error) => true,
	},

	setup(props, ctx) {
		const FileInputRef = ref<InstanceType<typeof ElInput>>();
		const filePath = ref('');
		const onPickFile = () => {
			FileInputRef.value?.$el.querySelector('input')?.click();
		};
		const onUpload = (f: string) => {
			const fileInput = FileInputRef.value?.$el.querySelector('input');
			const file = fileInput.files[0];
			let confirm: Promise<any> = Promise.resolve();
			if (props.confirm === true) {
				confirm = ElMessageBox.confirm(
					`确定要上传'${file.name}'文件吗？`,
					'提示'
				);
			} else if (props.confirm instanceof Promise) {
				confirm = props.confirm;
			}
			confirm
				.then(() => {
					let formData = new FormData();
					formData.append('file', file);
					ctx.emit('uploadBefore', formData, file);
					//if (props.uploadBefore) {
					//  formData = props.uploadBefore(formData, file)
					//}
					const api = props.uploadApi;
					api(formData)
						.then((res) => {
							ctx.emit('success', res);
						})
						.catch((err) => {
							ctx.emit('error', err);
							//ElMessageBox.alert(`上传'${file.name}'文件失败`, '提示')
						});
				})
				.catch(() => {
					ElMessageBox.alert(`取消上传'${file.name}'文件`, '提示');
				})
				.finally(() => {
					filePath.value = '';
				});
		};

		return () => (
			<>
				<ElButton
					showOverflowTooltip
					onClick={onPickFile}
					icon={UploadFilled}
					v-slots={{
						default: () => (
							<>
								{props.txt}
								<ElInput
									type="file"
									v-model={filePath.value}
									onChange={onUpload}
									ref={FileInputRef}
									accept={props.accept}
									style={{ display: 'none' }}
								/>
							</>
						),
					}}
				/>
			</>
		);
	},
});
