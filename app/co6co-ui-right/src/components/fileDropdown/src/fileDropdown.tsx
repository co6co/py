import { defineComponent, ref, SlotsType, VNode } from 'vue';
import {
	ElButton,
	ElDropdown,
	ElIcon,
	ElDropdownMenu,
	ElDropdownItem,
} from 'element-plus';
import {
	ArrowDown,
	Files as FilesIcon,
	Folder as FolderIcon,
} from '@element-plus/icons-vue';
import { IFileOption } from '@/constants';
export default defineComponent({
	name: 'FileSelect',
	props: {
		lable: {
			type: String,
			default: '上传文件或目录',
		},
		fileMenuDisabled: {
			type: Boolean,
			default: false,
		},
		FolderMenuDisabled: {
			type: Boolean,
			default: false,
		},
		fileMenuText: {
			type: String,
			default: '上传文件',
		},
		fileMultiple: {
			type: Boolean,
			default: true,
		},
		folterMenuText: {
			type: String,
			default: '上传文件夹',
		},
	},
	slots: Object as SlotsType<{
		items: () => any;
	}>,
	emits: {
		//@ts-ignore
		selected: (files: IFileOption[]) => true,
	},
	setup(prop, ctx) {
		/**选择上传 */
		const fileInputRef = ref();
		const onFileSelect = (event) => {
			const files: File[] = Array.from(event.target.files);
			const newData = files.map((file) => {
				const o = { file: file, finished: false };
				return o;
			});
			ctx.emit('selected', newData);
			//DATA.files.push(...newData);
		};
		//文件夹
		const folderInputRef = ref();
		const onFolderSelect = (event) => {
			const files: File[] = Array.from(event.target.files);
			const newData = files.map((file) => {
				const subPath = file.webkitRelativePath.replace(`/${file.name}`, '');
				const o = { file: file, finished: false, subPath: subPath };
				return o;
			});
			ctx.emit('selected', newData);
			//DATA.files.push(...newData);
		};
		const onCommand = (command: string) => {
			switch (command) {
				case 'file':
					fileInputRef.value.click();
					break;
				case 'folder':
					folderInputRef.value.click();
					break;
			}
		};
		const rander = (): VNode => {
			return (
				<ElDropdown trigger="click" onCommand={onCommand}>
					{{
						default: () => (
							<ElButton type="primary">
								{prop.lable}
								<ElIcon>
									<ArrowDown />
								</ElIcon>
							</ElButton>
						),
						dropdown: () => (
							<ElDropdownMenu>
								<ElDropdownItem
									disabled={prop.fileMenuDisabled}
									icon={FilesIcon}
									command="file">
									{prop.fileMenuText}
									<input
										type="file"
										multiple
										ref={fileInputRef}
										style="display: none"
										onChange={onFileSelect}
									/>
								</ElDropdownItem>
								<ElDropdownItem
									disabled={prop.fileMenuDisabled}
									icon={FolderIcon}
									command="folder">
									{prop.folterMenuText}
									<input
										type="file"
										multiple
										ref={folderInputRef}
										webkitdirectory
										style="display: none"
										onChange={onFolderSelect}
									/>
								</ElDropdownItem>
								{ctx.slots.items?.()}
							</ElDropdownMenu>
						),
					}}
				</ElDropdown>
			);
		};
		/**
		ctx.expose({
			openDialog,
		});
		rander.openDialog = openDialog;
		 */
		return rander;
	}, //end setup
});
