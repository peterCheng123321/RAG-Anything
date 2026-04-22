"""
Batch processing functionality for RAGAnything

Contains methods for processing multiple documents in batch mode
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import time

from .batch_parser import BatchParser, BatchProcessingResult

if TYPE_CHECKING:
    from .config import RAGAnythingConfig


class BatchMixin:
    """BatchMixin class containing batch processing functionality for RAGAnything"""

    # Type hints for mixin attributes (will be available when mixed into RAGAnything)
    config: "RAGAnythingConfig"
    logger: logging.Logger

    # Type hints for methods that will be available from other mixins
    async def _ensure_lightrag_initialized(self) -> None: ...
    async def process_document_complete(self, file_path: str, **kwargs) -> None: ...

    # ==========================================
    # ORIGINAL BATCH PROCESSING METHOD (RESTORED)
    # ==========================================

    async def process_folder_complete(
        self,
        folder_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = None,
        max_workers: int = None,
    ):
        """
        Process all supported files in a folder

        Args:
            folder_path: Path to the folder containing files to process
            output_dir: Directory for parsed outputs (optional)
            parse_method: Parsing method to use (optional)
            display_stats: Whether to display statistics (optional)
            split_by_character: Character to split by (optional)
            split_by_character_only: Whether to split only by character (optional)
            file_extensions: List of file extensions to process (optional)
            recursive: Whether to process folders recursively (optional)
            max_workers: Maximum number of workers for concurrent processing (optional)
        """
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = True
        if file_extensions is None:
            file_extensions = self.config.supported_file_extensions
        if recursive is None:
            recursive = self.config.recursive_folder_processing
        if max_workers is None:
            max_workers = self.config.max_concurrent_files

        await self._ensure_lightrag_initialized()

        # Get all files in the folder
        folder_path_obj = Path(folder_path)
        if not folder_path_obj.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Collect files based on supported extensions
        files_to_process = []
        for file_ext in file_extensions:
            if recursive:
                pattern = f"**/*{file_ext}"
            else:
                pattern = f"*{file_ext}"
            files_to_process.extend(folder_path_obj.glob(pattern))

        if not files_to_process:
            self.logger.warning(f"No supported files found in {folder_path}")
            return

        self.logger.info(
            f"Found {len(files_to_process)} files to process in {folder_path}"
        )

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process files with controlled concurrency
        semaphore = asyncio.Semaphore(max_workers)
        tasks = []

        async def process_single_file(file_path: Path):
            async with semaphore:
                is_in_subdir = (
                    lambda file_path, dir_path: len(
                        file_path.relative_to(dir_path).parents
                    )
                    > 1
                )(file_path, folder_path_obj)

                try:
                    await self.process_document_complete(
                        str(file_path),
                        output_dir=(
                            output_dir
                            if not is_in_subdir
                            else str(
                                output_path
                                / file_path.parent.relative_to(folder_path_obj)
                            )
                        ),
                        parse_method=parse_method,
                        split_by_character=split_by_character,
                        split_by_character_only=split_by_character_only,
                        file_name=(
                            None
                            if not is_in_subdir
                            else str(file_path.relative_to(folder_path_obj))
                        ),
                    )
                    return True, str(file_path), None
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
                    return False, str(file_path), str(e)

        # Create tasks for all files
        for file_path in files_to_process:
            task = asyncio.create_task(process_single_file(file_path))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_files = []
        failed_files = []
        for result in results:
            if isinstance(result, Exception):
                failed_files.append(("unknown", str(result)))
            else:
                success, file_path, error = result
                if success:
                    successful_files.append(file_path)
                else:
                    failed_files.append((file_path, error))

        # Display statistics if requested
        if display_stats:
            self.logger.info("Processing complete!")
            self.logger.info(f"  Successful: {len(successful_files)} files")
            self.logger.info(f"  Failed: {len(failed_files)} files")
            if failed_files:
                self.logger.warning("Failed files:")
                for file_path, error in failed_files:
                    self.logger.warning(f"  - {file_path}: {error}")

    # ==========================================
    # NEW ENHANCED BATCH PROCESSING METHODS
    # ==========================================

    def process_documents_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> BatchProcessingResult:
        """
        Process multiple documents in batch using the new BatchParser

        Args:
            file_paths: List of file paths or directories to process
            output_dir: Output directory for parsed files
            parse_method: Parsing method to use
            max_workers: Maximum number of workers for parallel processing
            recursive: Whether to process directories recursively
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments passed to the parser

        Returns:
            BatchProcessingResult: Results of the batch processing
        """
        # Use config defaults if not specified
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        # Create batch parser
        batch_parser = BatchParser(
            parser_type=self.config.parser,
            max_workers=max_workers,
            show_progress=show_progress,
            skip_installation_check=True,  # Skip installation check for better UX
        )

        # Process batch
        return batch_parser.process_batch(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            recursive=recursive,
            **kwargs,
        )

    async def process_documents_batch_async(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> BatchProcessingResult:
        """
        Asynchronously process multiple documents in batch

        Args:
            file_paths: List of file paths or directories to process
            output_dir: Output directory for parsed files
            parse_method: Parsing method to use
            max_workers: Maximum number of workers for parallel processing
            recursive: Whether to process directories recursively
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments passed to the parser

        Returns:
            BatchProcessingResult: Results of the batch processing
        """
        # Use config defaults if not specified
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        # Create batch parser
        batch_parser = BatchParser(
            parser_type=self.config.parser,
            max_workers=max_workers,
            show_progress=show_progress,
            skip_installation_check=True,  # Skip installation check for better UX
        )

        # Process batch asynchronously
        return await batch_parser.process_batch_async(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            recursive=recursive,
            **kwargs,
        )

    def get_supported_file_extensions(self) -> List[str]:
        """Get list of supported file extensions for batch processing"""
        batch_parser = BatchParser(parser_type=self.config.parser)
        return batch_parser.get_supported_extensions()

    def filter_supported_files(
        self, file_paths: List[str], recursive: Optional[bool] = None
    ) -> List[str]:
        """
        Filter file paths to only include supported file types

        Args:
            file_paths: List of file paths to filter
            recursive: Whether to process directories recursively

        Returns:
            List of supported file paths
        """
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        batch_parser = BatchParser(parser_type=self.config.parser)
        return batch_parser.filter_supported_files(file_paths, recursive)

    # ==========================================
    # INCREMENTAL FOLDER PROCESSING
    # ==========================================

    @staticmethod
    def _file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
        """Return the hex MD5 digest of *path* without loading it entirely into RAM."""
        h = hashlib.md5()
        with open(path, "rb") as fh:
            while True:
                block = fh.read(chunk_size)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _load_scan_state(state_file: Path) -> Dict[str, Dict[str, Any]]:
        """Load incremental scan state from *state_file*.

        Returns a mapping of ``str(absolute_path)`` → ``{"mtime": float, "size": int, "md5": str}``.
        """
        if not state_file.exists():
            return {}
        try:
            with open(state_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    @staticmethod
    def _save_scan_state(state_file: Path, state: Dict[str, Dict[str, Any]]) -> None:
        """Atomically persist the scan state to *state_file*."""
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
            os.replace(tmp, state_file)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _is_file_changed(
        self,
        file_path: Path,
        state: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Return ``True`` when *file_path* is new or its content has changed.

        The check uses a two-stage approach for speed:

        1. Compare modification-time *and* size against stored values — if
           both match the file is almost certainly identical and we skip the
           expensive MD5 pass.
        2. If mtime or size differ, compute the MD5 and compare against the
           stored digest.  This catches the rare case where a file is
           recreated with the same timestamp but different content.
        """
        key = str(file_path.resolve())
        if key not in state:
            return True  # never seen before

        entry = state[key]
        try:
            stat = file_path.stat()
        except OSError:
            return True  # can't stat → assume changed

        # Fast path: mtime + size unchanged → skip MD5
        if (
            entry.get("mtime") == stat.st_mtime
            and entry.get("size") == stat.st_size
        ):
            return False

        # Slow path: compute MD5
        try:
            current_md5 = self._file_md5(file_path)
        except OSError:
            return True
        return current_md5 != entry.get("md5", "")

    def _record_file(
        self,
        file_path: Path,
        state: Dict[str, Dict[str, Any]],
    ) -> None:
        """Update *state* with the current mtime, size, and MD5 of *file_path*."""
        key = str(file_path.resolve())
        try:
            stat = file_path.stat()
            md5 = self._file_md5(file_path)
            state[key] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "md5": md5,
            }
        except OSError:
            pass  # if we can't stat/hash, just don't update the record

    async def process_folder_incremental(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        display_stats: bool = True,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
        file_extensions: Optional[List[str]] = None,
        recursive: Optional[bool] = None,
        max_workers: Optional[int] = None,
        state_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process only new or changed files in a folder (incremental scan).

        Works like :meth:`process_folder_complete` but maintains a persistent
        state file that records each file's modification time, size, and MD5
        digest.  On subsequent runs only files whose content has changed since
        the last successful processing are re-indexed, making large-folder
        workflows much faster.

        Args:
            folder_path: Path to the folder containing files to process.
            output_dir: Directory for parsed outputs.
            parse_method: Parsing method to use.
            display_stats: Whether to log a summary after processing.
            split_by_character: Character to split text chunks on.
            split_by_character_only: Split exclusively by *split_by_character*.
            file_extensions: File extensions to consider (e.g. ``[".pdf", ".txt"]``).
            recursive: Whether to descend into sub-folders.
            max_workers: Maximum number of files processed concurrently.
            state_file: Path to the JSON state file.  Defaults to
                ``<working_dir>/incremental_scan_state.json``.

        Returns:
            A dict with keys ``skipped``, ``processed``, ``failed``, and
            ``state_file`` (path to the persisted state).
        """
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if file_extensions is None:
            file_extensions = self.config.supported_file_extensions
        if recursive is None:
            recursive = self.config.recursive_folder_processing
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if state_file is None:
            state_file = str(
                Path(self.config.working_dir) / "incremental_scan_state.json"
            )

        await self._ensure_lightrag_initialized()

        folder_path_obj = Path(folder_path)
        if not folder_path_obj.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        state_path = Path(state_file)
        state = self._load_scan_state(state_path)

        # Collect candidate files
        files_to_consider: List[Path] = []
        for ext in file_extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files_to_consider.extend(folder_path_obj.glob(pattern))

        if not files_to_consider:
            self.logger.warning(f"No supported files found in {folder_path}")
            return {"skipped": 0, "processed": 0, "failed": 0, "state_file": str(state_path)}

        # Split into changed / unchanged
        files_to_process = [f for f in files_to_consider if self._is_file_changed(f, state)]
        skipped_count = len(files_to_consider) - len(files_to_process)

        self.logger.info(
            f"Incremental scan of {folder_path}: "
            f"{len(files_to_consider)} total, "
            f"{len(files_to_process)} changed/new, "
            f"{skipped_count} unchanged (skipped)"
        )

        if not files_to_process:
            if display_stats:
                self.logger.info(
                    f"Nothing to do — all {skipped_count} file(s) are up to date."
                )
            return {
                "skipped": skipped_count,
                "processed": 0,
                "failed": 0,
                "state_file": str(state_path),
            }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_workers)
        # Protect concurrent writes to the shared state dict + state file
        state_lock = asyncio.Lock()

        async def process_single_file(file_path: Path):
            async with semaphore:
                is_in_subdir = len(file_path.relative_to(folder_path_obj).parents) > 1
                try:
                    await self.process_document_complete(
                        str(file_path),
                        output_dir=(
                            output_dir
                            if not is_in_subdir
                            else str(output_path / file_path.parent.relative_to(folder_path_obj))
                        ),
                        parse_method=parse_method,
                        split_by_character=split_by_character,
                        split_by_character_only=split_by_character_only,
                        file_name=(
                            None
                            if not is_in_subdir
                            else str(file_path.relative_to(folder_path_obj))
                        ),
                    )
                    # Persist state immediately after each success
                    async with state_lock:
                        self._record_file(file_path, state)
                        self._save_scan_state(state_path, state)
                    return True, str(file_path), None
                except Exception as exc:
                    self.logger.error(f"Failed to process {file_path}: {exc}")
                    return False, str(file_path), str(exc)

        tasks = [asyncio.create_task(process_single_file(fp)) for fp in files_to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_files: List[str] = []
        failed_files: List[tuple] = []
        for result in results:
            if isinstance(result, Exception):
                failed_files.append(("unknown", str(result)))
            else:
                success, fp, error = result
                if success:
                    successful_files.append(fp)
                else:
                    failed_files.append((fp, error))

        if display_stats:
            self.logger.info("Incremental processing complete!")
            self.logger.info(f"  Skipped (unchanged): {skipped_count}")
            self.logger.info(f"  Processed:           {len(successful_files)}")
            self.logger.info(f"  Failed:              {len(failed_files)}")
            if failed_files:
                self.logger.warning("Failed files:")
                for fp, err in failed_files:
                    self.logger.warning(f"  - {fp}: {err}")

        return {
            "skipped": skipped_count,
            "processed": len(successful_files),
            "failed": len(failed_files),
            "state_file": str(state_path),
        }

    async def process_documents_with_rag_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process documents in batch and then add them to RAG

        This method combines document parsing and RAG insertion:
        1. First, parse all documents using batch processing
        2. Then, process each successfully parsed document with RAG

        Args:
            file_paths: List of file paths or directories to process
            output_dir: Output directory for parsed files
            parse_method: Parsing method to use
            max_workers: Maximum number of workers for parallel processing
            recursive: Whether to process directories recursively
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments passed to the parser

        Returns:
            Dict containing both parse results and RAG processing results
        """
        start_time = time.time()
        callback_manager = getattr(self, "callback_manager", None)
        total_files = len(file_paths)

        if callback_manager is not None:
            callback_manager.dispatch(
                "on_batch_start",
                file_count=total_files,
            )

        # Use config defaults if not specified
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        self.logger.info("Starting batch processing with RAG integration")

        # Step 1: Parse documents in batch
        parse_result = self.process_documents_batch(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            max_workers=max_workers,
            recursive=recursive,
            show_progress=show_progress,
            **kwargs,
        )

        # Step 2: Process with RAG
        # Initialize RAG system
        await self._ensure_lightrag_initialized()

        # Then, process each successful file with RAG
        rag_results = {}

        if parse_result.successful_files:
            self.logger.info(
                f"Processing {len(parse_result.successful_files)} files with RAG"
            )

            # Process files with RAG (this could be parallelized in the future)
            for file_path in parse_result.successful_files:
                try:
                    # Process the successfully parsed file with RAG
                    await self.process_document_complete(
                        file_path,
                        output_dir=output_dir,
                        parse_method=parse_method,
                        **kwargs,
                    )

                    # Get some statistics about the processed content
                    # This would require additional tracking in the RAG system
                    rag_results[file_path] = {"status": "success", "processed": True}

                except Exception as e:
                    self.logger.error(
                        f"Failed to process {file_path} with RAG: {str(e)}"
                    )
                    rag_results[file_path] = {
                        "status": "failed",
                        "error": str(e),
                        "processed": False,
                    }

        processing_time = time.time() - start_time

        successful_rag_files = len([r for r in rag_results.values() if r["processed"]])
        failed_rag_files = len([r for r in rag_results.values() if not r["processed"]])

        if callback_manager is not None:
            callback_manager.dispatch(
                "on_batch_complete",
                total_files=total_files,
                successful=successful_rag_files,
                failed=failed_rag_files,
                duration_seconds=processing_time,
            )

        return {
            "parse_result": parse_result,
            "rag_results": rag_results,
            "total_processing_time": processing_time,
            "successful_rag_files": successful_rag_files,
            "failed_rag_files": failed_rag_files,
        }
