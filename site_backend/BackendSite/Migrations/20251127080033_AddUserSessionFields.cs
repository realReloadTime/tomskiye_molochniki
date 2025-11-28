using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace BackendSite.Migrations
{
    /// <inheritdoc />
    public partial class AddUserSessionFields : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<DateTime>(
                name: "SessionExpires",
                table: "Users",
                type: "datetime2",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "SessionToken",
                table: "Users",
                type: "nvarchar(max)",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "SessionExpires",
                table: "Users");

            migrationBuilder.DropColumn(
                name: "SessionToken",
                table: "Users");
        }
    }
}
