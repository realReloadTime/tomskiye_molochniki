Инструкция по запуску .NET бэкенда
Предварительные требования
.NET 9 SDK
SQL Server (Express edition достаточно)

Создание базы данных и миграции
1. dotnet restore
2. dotnet ef migrations add InitialCreate
3. dotnet ef database update

Запуск приложения
dotnet run
