Инструкция по запуску .NET бэкенда
Предварительные требования
.NET 9 SDK, 
SQL Server (Express edition достаточно)

(в консоли проекта BackendSite):
1 Создание базы данных и миграции
  1. dotnet restore
  2. dotnet ef migrations add InitialCreate
  3. dotnet ef database update

2. Запуск приложения
dotnet run
