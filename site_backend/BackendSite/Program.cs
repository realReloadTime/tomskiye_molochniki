using Scalar.AspNetCore;
using Microsoft.EntityFrameworkCore;
using BackendSite.Models;
using BackendSite.Data;
using BackendSite.Services; 

var builder = WebApplication.CreateBuilder(args);


builder.Services.AddOpenApi();
builder.Services.AddControllers();


builder.Services.AddDbContext<ReviewsHackatonDBContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddScoped<IPasswordHasherService, PasswordHasherService>();

var app = builder.Build();


app.UseRouting();
app.UseAuthorization();

app.MapControllers();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.MapScalarApiReference();
}

app.UseHttpsRedirection();

app.Run();