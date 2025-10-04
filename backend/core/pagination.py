"""
Система пагинации для ExoplanetAI API
"""
from typing import Generic, List, Optional, TypeVar
from math import ceil

from fastapi import Query
from pydantic import BaseModel, Field

from core.constants import APIConstants

T = TypeVar('T')


class PaginationParams(BaseModel):
    """Параметры пагинации"""
    page: int = Field(
        default=1, 
        ge=1, 
        description="Номер страницы (начиная с 1)"
    )
    page_size: int = Field(
        default=APIConstants.DEFAULT_PAGE_SIZE,
        ge=APIConstants.MIN_PAGE_SIZE,
        le=APIConstants.MAX_PAGE_SIZE,
        description=f"Размер страницы ({APIConstants.MIN_PAGE_SIZE}-{APIConstants.MAX_PAGE_SIZE})"
    )
    
    @property
    def skip(self) -> int:
        """Количество записей для пропуска"""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Лимит записей на странице"""
        return self.page_size


class PaginatedResponse(BaseModel, Generic[T]):
    """Стандартный ответ с пагинацией"""
    items: List[T] = Field(description="Элементы текущей страницы")
    total: int = Field(description="Общее количество элементов")
    page: int = Field(description="Текущая страница")
    page_size: int = Field(description="Размер страницы")
    total_pages: int = Field(description="Общее количество страниц")
    has_next: bool = Field(description="Есть ли следующая страница")
    has_prev: bool = Field(description="Есть ли предыдущая страница")
    next_page: Optional[int] = Field(description="Номер следующей страницы")
    prev_page: Optional[int] = Field(description="Номер предыдущей страницы")
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        params: PaginationParams
    ) -> 'PaginatedResponse[T]':
        """
        Создание пагинированного ответа
        
        Args:
            items: Элементы текущей страницы
            total: Общее количество элементов
            params: Параметры пагинации
            
        Returns:
            Пагинированный ответ
        """
        total_pages = ceil(total / params.page_size) if params.page_size > 0 else 0
        has_next = params.page < total_pages
        has_prev = params.page > 1
        
        return cls(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev,
            next_page=params.page + 1 if has_next else None,
            prev_page=params.page - 1 if has_prev else None
        )


def get_pagination_params(
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(
        APIConstants.DEFAULT_PAGE_SIZE,
        ge=APIConstants.MIN_PAGE_SIZE,
        le=APIConstants.MAX_PAGE_SIZE,
        description="Размер страницы"
    )
) -> PaginationParams:
    """Dependency для получения параметров пагинации"""
    return PaginationParams(page=page, page_size=page_size)


class SearchPaginationParams(PaginationParams):
    """Расширенные параметры пагинации для поиска"""
    sort_by: str = Field(
        default="relevance",
        description="Поле для сортировки"
    )
    sort_order: str = Field(
        default="desc",
        pattern="^(asc|desc)$",
        description="Порядок сортировки (asc/desc)"
    )
    
    
def get_search_pagination_params(
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(
        APIConstants.DEFAULT_PAGE_SIZE,
        ge=APIConstants.MIN_PAGE_SIZE,
        le=APIConstants.MAX_PAGE_SIZE,
        description="Размер страницы"
    ),
    sort_by: str = Query("relevance", description="Поле для сортировки"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Порядок сортировки")
) -> SearchPaginationParams:
    """Dependency для получения параметров пагинации поиска"""
    return SearchPaginationParams(
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order
    )
