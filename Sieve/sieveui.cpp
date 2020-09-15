//******************************************************************************
// Copyright 2020 ThirtySomething
//******************************************************************************
// This file is part of Sieve.
//
// Sieve is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// Sieve is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
// more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Sieve. If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#include "sieveui.h"
#include "ui_sieveui.h"
#include <QFileDialog>
#include <QMessageBox>

SieveUI::SieveUI(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::SieveUI)
    , m_sieve(new net::derpaul::sieve::CSieve(0LL))
{
    ui->setupUi(this);
}

SieveUI::~SieveUI()
{
    delete ui;
    delete m_sieve;
}


void SieveUI::on_actionQuit_triggered()
{
    QCoreApplication::quit();
}

void SieveUI::on_actionLoad_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(
        this,
        tr("Load prime data"), "",
        tr("Prime data (*.pd);;All Files (*)")
    );

    m_sieve->dataSave(fileName.toStdString());
}

void SieveUI::on_actionSave_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(
        this,
        tr("Save prime data"), "",
        tr("Prime data (*.pd);;All Files (*)")
    );

    m_sieve->dataSave(fileName.toStdString());
}

void SieveUI::on_actionAbout_Sieve_triggered()
{
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("About Sieve");
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setTextFormat(Qt::RichText);
    msgBox.setText("<a href='https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes'>Sieve of Eratosthenes</a><br>(C) 2020 by <a href='https://github.com/ThirtySomething/Sieve'>ThirtySomething</a>");
    msgBox.exec();
}
